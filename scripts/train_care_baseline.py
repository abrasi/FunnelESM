#!/usr/bin/env python3
"""Train one CARE model: internal-validation pilot or final full-train refit."""

import argparse
import gc
import importlib.metadata
import json
import math
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from train_deeploc_baseline import aggregate_resources, model_input_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from funnel_esm.care import CARE_LABELS, load_care
from funnel_esm.metrics import (
    apply_thresholds,
    compute_care_metrics,
    inverse_frequency_label_weights,
    sigmoid,
)
from funnel_esm.modeling_classification import (
    MaskedAttentionEsmForSequenceClassification,
    load_pretrained_classifier,
)


def load_model(args):
    return load_pretrained_classifier(
        MaskedAttentionEsmForSequenceClassification,
        args.model_name,
        CARE_LABELS,
        args.focal_gamma,
    )


def measure_phase(monitor, name, operation):
    """Use the same GPU aggregation as the DeepLoc trainer."""
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    monitor.begin_window(name)
    try:
        output = operation()
    finally:
        measurement = monitor.end_window(name)
    energy, duration, _ = aggregate_resources(measurement.total_energy, measurement.time, 0.0)
    if not math.isfinite(energy) or energy <= 0 or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Zeus returned an invalid energy/time measurement")
    return output, {
        "gpu_energy_j": energy,
        "gpu_energy_kwh": energy / 3_600_000,
        "time_s": duration,
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_predictions(path, protein_ids, output):
    import pandas as pd

    probabilities = sigmoid(output.predictions)
    labels = np.asarray(output.label_ids, dtype=np.int64)
    predictions = apply_thresholds(
        probabilities, np.full(7, 0.5), force_at_least_one=False, inclusive=False
    )
    frame = {"protein_id": protein_ids, "predicted_ec1": np.argmax(probabilities, axis=1) + 1}
    for index, label in enumerate(CARE_LABELS):
        frame[f"prob_{label}"] = probabilities[:, index]
        frame[f"true_{label}"] = labels[:, index]
        frame[f"pred_{label}"] = predictions[:, index]
    pd.DataFrame(frame).to_csv(path, index=False)


def train_run(args, model_loader=load_model):
    pilot = args.convergence_pilot
    output = Path(args.output_dir)
    if (output / "run_result.json").exists() or (output / "final_model").exists():
        raise ValueError("A completed run already exists; use a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data = load_care(
        args.train_csv,
        tokenizer,
        pilot=pilot,
        test_csvs=args.test_csvs,
        cluster_metadata_csv=args.cluster_metadata_csv,
        max_residues=args.max_residues,
    )
    model = model_loader(args)
    model.config.funnel_architecture = args.architecture
    model.config.care_ec1_policy = "all"
    weights = inverse_frequency_label_weights(data["train"]["labels"])
    model.set_label_weights(torch.tensor(weights, dtype=torch.float32))
    training_args = TrainingArguments(
        output_dir=str(output),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch" if pilot else "no",
        logging_strategy="epoch",
        save_strategy="no",
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    inputs = {name: model_input_dataset(dataset) for name, dataset in data.items()}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs["train"],
        eval_dataset=inputs.get("validation"),
        compute_metrics=compute_care_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    main_process = trainer.is_world_process_zero()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if main_process:
        save_json(
            output / "training_protocol.json",
            {
                "phase": "pilot" if pilot else "final",
                "training_arguments": training_args.to_dict(),
                "model_name": args.model_name,
                "architecture": args.architecture,
                "pool_after_layers": getattr(model.config, "funnel_pool_after_layers", []),
                "backbone_frozen": True,
                "ec1_policy": "all",
                "focal_gamma": 1.0,
                "training_examples": len(data["train"]),
                "validation_examples": len(data["validation"]) if pilot else 0,
                "validation": {
                    "size": 0.1,
                    "seed": 0,
                    "grouping": "clusterRes50" if args.cluster_metadata_csv else "sequence",
                }
                if pilot
                else None,
                "effective_train_batch_size": training_args.train_batch_size
                * world_size
                * args.gradient_accumulation_steps,
                "selection": "manual LR/epochs; no automatic checkpoint selection; no test-based selection",
                "runtime": {
                    "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "cuda": torch.version.cuda,
                    "world_size": world_size,
                },
            },
        )

    # Each torchrun process monitors its own GPU; energy is summed above.
    from zeus.monitor import ZeusMonitor

    gpu_indices = (
        [torch.cuda.current_device()] if dist.is_initialized() else list(range(training_args.n_gpu))
    )
    monitor = ZeusMonitor(gpu_indices=gpu_indices, cpu_indices=[], sync_execution_with="torch")
    for gpu in gpu_indices:
        torch.cuda.reset_peak_memory_stats(gpu)
    _, training = measure_phase(monitor, "care_training", trainer.train)
    peak = (
        max(torch.cuda.max_memory_allocated(gpu) for gpu in gpu_indices) / 1_000_000_000
        if gpu_indices
        else None
    )
    if dist.is_initialized():
        peak_tensor = torch.tensor(peak, dtype=torch.float64, device=training_args.device)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        peak = peak_tensor.item()
    result = {
        "phase": "pilot" if pilot else "final_full_train_refit",
        "architecture": args.architecture,
        "model_name": args.model_name,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "training_examples": len(data["train"]),
        "pool_after_layers": getattr(model.config, "funnel_pool_after_layers", []),
        "resources": {
            "train_gpu_energy_j": training["gpu_energy_j"],
            "train_gpu_energy_kwh": training["gpu_energy_kwh"],
            "train_time_s": training["time_s"],
            "peak_gpu_memory_gb": peak,
        },
    }
    if main_process:
        save_json(output / "epoch_history.json", trainer.state.log_history)

    if pilot:
        history = [row for row in trainer.state.log_history if "eval_loss" in row]
        best = min(history, key=lambda row: row["eval_loss"])
        result.update(
            best_epoch=best["epoch"],
            best_validation_loss=best["eval_loss"],
            best_validation_metrics={
                key: value for key, value in best.items() if key.startswith("eval_")
            },
        )
    else:
        result["test_metrics"] = {}
        result["resources"]["test_by_split"] = {}
        for path in args.test_csvs:
            name = Path(path).stem
            prediction, resources = measure_phase(
                monitor,
                f"care_test_{name}",
                partial(trainer.predict, inputs[name], metric_key_prefix="test"),
            )
            result["test_metrics"][name] = {
                "n_examples": len(data[name]),
                "metrics": {
                    key.removeprefix("test_"): float(value)
                    for key, value in prediction.metrics.items()
                    if key.startswith("test_") and isinstance(value, (int, float, np.number))
                },
            }
            result["resources"]["test_by_split"][name] = resources
            if main_process:
                save_predictions(
                    output / f"{name}_predictions.csv", data[name]["protein_id"], prediction
                )
        if main_process:
            # Save current ESM keys; reversing the pretrained RoPE mapping would
            # write a literal wildcard key that our strict evaluator cannot load.
            trainer.save_model(str(output / "final_model"))
            tokenizer.save_pretrained(output / "final_model")
    if main_process:
        save_json(output / "run_result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
    del trainer, model, data, monitor
    gc.collect()
    torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csvs", nargs="+", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument(
        "--architecture", default="baseline", help="Experiment name recorded in outputs"
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max_residues", type=int, default=1022)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--cluster_metadata_csv")
    parser.add_argument("--convergence_pilot", action="store_true")
    return parser


def run(args, model_loader=load_model):
    if (
        args.epochs < 1
        or args.learning_rate <= 0
        or args.batch_size < 1
        or args.gradient_accumulation_steps < 1
    ):
        raise ValueError("epochs, learning_rate, batch_size and accumulation must be positive")
    if args.convergence_pilot and args.test_csvs:
        raise ValueError("The pilot must not load official tests")
    if not args.convergence_pilot and not args.test_csvs:
        raise ValueError("Final runs require --test_csvs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Zeus measurement")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --mixed_precision no")
    train_run(args, model_loader=model_loader)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
