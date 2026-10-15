#!/usr/bin/env python3
"""Fit one CARE model on all official training data and evaluate test splits."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from zeus.monitor import ZeusMonitor

from funnel_esm.care import CARE_LABELS, load_care
from funnel_esm.metrics import (
    apply_thresholds,
    inverse_frequency_label_weights,
    sigmoid,
)
from train_care import (
    aggregate_resources,
    build_model,
    compute_care_metrics,
    model_input_dataset,
    package_version,
    save_test_predictions,
)


def parser():
    result = argparse.ArgumentParser(
        description=(
            "Train one CARE architecture on the complete official training "
            "set, then evaluate the same model on multiple official test splits."
        )
    )
    result.add_argument("--train_csv", required=True)
    result.add_argument("--test_csvs", nargs="+", required=True)
    result.add_argument("--output_dir", required=True)
    result.add_argument("--architecture", required=True)
    result.add_argument("--model_name", default="facebook/esm2_t12_35M_UR50D")
    result.add_argument("--batch_size", type=int, default=8)
    result.add_argument("--epochs", type=int, required=True)
    result.add_argument("--learning_rate", type=float, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--validation_seed", type=int, default=0)
    result.add_argument("--validation_size", type=float, default=0.1)
    result.add_argument("--max_residues", type=int, default=1022)
    result.add_argument("--cluster_metadata_csv")
    result.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    result.add_argument("--focal_gamma", type=float, default=1.0)
    return result


def main():
    args = parser().parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, learning_rate, and batch_size must be positive")
    if args.focal_gamma < 0:
        raise ValueError("focal_gamma cannot be negative")
    if not args.test_csvs:
        raise ValueError("Provide at least one official test CSV")
    test_paths = [Path(path).resolve() for path in args.test_csvs]
    if len({path.stem for path in test_paths}) != len(test_paths):
        raise ValueError("Test CSV basenames must be unique")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Zeus energy measurement")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} GPUs are visible"
            )
        torch.cuda.set_device(local_rank)
    if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU; choose another precision")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data = load_care(
        args.train_csv,
        tokenizer,
        test_csv=[str(path) for path in test_paths],
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
        max_residues=args.max_residues,
        cluster_metadata_csv=args.cluster_metadata_csv,
    )

    # threshold_calibration is the tokenized, grouped version of all official
    # training proteins. Use it as the actual final fit set after the pilot.
    full_train = data["threshold_calibration"]
    label_weights = inverse_frequency_label_weights(full_train["labels"])
    model_args = argparse.Namespace(**vars(args))
    model = build_model(model_args)
    model.set_label_weights(torch.tensor(label_weights, dtype=torch.float32))

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="no",
        logging_strategy="epoch",
        save_strategy="no",
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=model_input_dataset(full_train),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    is_main_process = trainer.is_world_process_zero()
    monitored_gpus = (
        [torch.cuda.current_device()]
        if dist.is_initialized()
        else list(range(torch.cuda.device_count()))
    )
    gpu_names = [torch.cuda.get_device_name(index) for index in monitored_gpus]
    for index in monitored_gpus:
        torch.cuda.reset_peak_memory_stats(index)

    if is_main_process:
        (output_dir / "training_protocol.json").write_text(
            json.dumps(
                {
                    "phase": "final_full_train_refit",
                    "training_arguments": training_args.to_dict(),
                    "training_examples": len(full_train),
                    "effective_train_batch_size": args.batch_size * world_size,
                    "selection": "hyperparameters and epoch budget fixed from internal pilot; no validation or test used during final fit",
                    "test_csvs": [str(path) for path in test_paths],
                    "test_used_for_selection": False,
                    "labels": list(CARE_LABELS),
                    "runtime": {
                        "torch": torch.__version__,
                        "transformers": package_version("transformers"),
                        "zeus": package_version("zeus", "zeus-ml"),
                        "cuda_runtime": torch.version.cuda,
                        "gpu": gpu_names,
                        "world_size": world_size,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    monitor = ZeusMonitor(gpu_indices=monitored_gpus, cpu_indices=[])
    monitor.begin_window("care_full_train")
    trainer.train()
    local_training = monitor.end_window("care_full_train")
    local_peak_memory_gb = max(
        torch.cuda.max_memory_allocated(index) for index in monitored_gpus
    ) / 1_000_000_000
    train_energy_j, train_time_s, peak_memory_gb = aggregate_resources(
        local_training.total_energy,
        local_training.time,
        local_peak_memory_gb,
    )

    trainer.compute_metrics = lambda prediction: compute_care_metrics(
        prediction,
        thresholds=np.full(len(CARE_LABELS), 0.5, dtype=np.float64),
        force_at_least_one=False,
        inclusive=False,
    )
    fixed_thresholds = np.full(len(CARE_LABELS), 0.5, dtype=np.float64)
    test_results = {}
    test_resource_results = {}
    predictions_dir = output_dir / "test_predictions"
    if is_main_process:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for index, test_path in enumerate(test_paths):
        key = test_path.stem
        dataset = model_input_dataset(data[f"test_{index}"])
        window = f"care_test_{index}"
        monitor.begin_window(window)
        test_output = trainer.predict(dataset, metric_key_prefix="test")
        local_test = monitor.end_window(window)
        test_energy_j, test_time_s, _ = aggregate_resources(
            local_test.total_energy,
            local_test.time,
            0.0,
        )
        probabilities = sigmoid(test_output.predictions)
        labels = np.asarray(test_output.label_ids, dtype=np.int64)
        predictions = apply_thresholds(
            probabilities,
            fixed_thresholds,
            force_at_least_one=False,
            inclusive=False,
        )
        test_metrics = {
            metric.removeprefix("test_"): float(value)
            for metric, value in test_output.metrics.items()
            if metric.startswith("test_")
            and isinstance(value, (float, int, np.number))
        }
        test_results[key] = {
            "test_csv": str(test_path),
            "n_examples": len(dataset),
            "metrics": test_metrics,
            "thresholds": {label: 0.5 for label in CARE_LABELS},
        }
        test_resource_results[key] = {
            "gpu_energy_j": float(test_energy_j),
            "gpu_energy_kwh": float(test_energy_j / 3_600_000),
            "time_s": float(test_time_s),
        }
        if is_main_process:
            save_test_predictions(
                predictions_dir / f"{key}_predictions.csv",
                data[f"test_{index}"]["protein_id"],
                probabilities,
                labels,
                predictions,
            )

    if is_main_process:
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(output_dir / "final_model")
        result = {
            "phase": "final_full_train_refit",
            "architecture": args.architecture,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "training_examples": len(full_train),
            "validation_examples_used_in_final_fit": 0,
            "test_metrics": test_results,
            "resources": {
                "train_gpu_energy_j": float(train_energy_j),
                "train_gpu_energy_kwh": float(train_energy_j / 3_600_000),
                "train_time_s": float(train_time_s),
                "peak_gpu_memory_gb": float(peak_memory_gb),
                "test_by_split": test_resource_results,
            },
        }
        (output_dir / "run_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
    del trainer, model, data, monitor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
