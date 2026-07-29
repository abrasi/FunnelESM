#!/usr/bin/env python3
"""Train and evaluate the frozen ESM-2 baseline with 5-fold cross-validation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from zeus.monitor import ZeusMonitor

from funnel_esm.data import DEEPLOC_LABELS, load_deeploc
from funnel_esm.metrics import (
    MCC_METRIC_NAMES,
    build_compute_metrics,
    inverse_frequency_label_weights,
    optimize_mcc_thresholds,
    sigmoid,
)
from funnel_esm.modeling_classification import (
    MaskedAttentionEsmForSequenceClassification,
)


ROPE_KEY_MAPPING = {
    "esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq":
        "esm.rotary_embeddings.inv_freq",
}


RESULT_NAMES = (
    "eval_loss",
    "eval_macro_f1",
    "eval_micro_f1",
    "eval_jaccard",
    "eval_accuracy",
    "eval_num_labels_predicted",
    "eval_macro_mcc",
    *tuple(f"eval_{name}" for name in MCC_METRIC_NAMES),
    "train_gpu_energy_j",
    "train_gpu_energy_kwh",
    "train_time_s",
    "threshold_calibration_gpu_energy_j",
    "threshold_calibration_gpu_energy_kwh",
    "threshold_calibration_time_s",
    "pipeline_gpu_energy_j",
    "pipeline_gpu_energy_kwh",
    "pipeline_time_s",
    "peak_gpu_memory_gb",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=1.0,
        help="Focal-loss gamma. DeepLoc 2.0 uses 1.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
        help="Precision shared by every architecture.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        default="facebook/esm2_t12_35M_UR50D",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        choices=range(5),
        default=None,
        metavar="FOLD",
        help=(
            "DeepLoc folds to run (0-4). By default, run all five folds."
        ),
    )
    parser.add_argument(
        "--eval_each_epoch",
        action="store_true",
        help=(
            "Evaluate after every epoch and save the resulting learning "
            "curve. Intended for convergence pilots."
        ),
    )
    parser.add_argument(
        "--convergence_pilot",
        action="store_true",
        help=(
            "After collecting epoch curves, skip threshold calibration and "
            "final model saving. Intended for LR/epoch selection only."
        ),
    )
    return parser


def parse_args():
    return build_parser().parse_args()


def load_model(args):
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(DEEPLOC_LABELS)
    config.problem_type = "multi_label_classification"
    config.deeploc_focal_gamma = args.focal_gamma
    config.deeploc_gaussian_kernel_size = 5
    # Five samples spanning [-sigma, sigma], following DeepLoc's
    # "size 5 clipped at one standard deviation" description.
    config.deeploc_gaussian_sigma = 2.0
    config.id2label = dict(enumerate(DEEPLOC_LABELS))
    config.label2id = {
        label: index for index, label in enumerate(DEEPLOC_LABELS)
    }

    return MaskedAttentionEsmForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        key_mapping=ROPE_KEY_MAPPING,
    )


def aggregate_resources(energy_j, time_s, memory_gb):
    """Aggregate one local GPU measurement across torchrun processes."""
    if not dist.is_initialized():
        return energy_j, time_s, memory_gb

    device = torch.device("cuda", torch.cuda.current_device())

    energy = torch.tensor(energy_j, dtype=torch.float64, device=device)
    dist.all_reduce(energy, op=dist.ReduceOp.SUM)

    run_time = torch.tensor(time_s, dtype=torch.float64, device=device)
    dist.all_reduce(run_time, op=dist.ReduceOp.MAX)

    memory = torch.tensor(memory_gb, dtype=torch.float64, device=device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    return energy.item(), run_time.item(), memory.item()


def train_fold(args, tokenizer, fold, model_loader=load_model):
    """Train one fold and return its metrics on the main process."""
    fold_seed = args.seed + fold
    set_seed(fold_seed)
    fold_output_dir = Path(args.output_dir) / f"fold_{fold}"

    datasets = load_deeploc(
        args.dataset_path,
        tokenizer,
        validation_partition=fold,
    )
    label_weights = inverse_frequency_label_weights(
        datasets["train"]["labels"]
    )

    model = model_loader(args)
    model.set_label_weights(
        torch.tensor(label_weights, dtype=torch.float32)
    )

    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch" if args.eval_each_epoch else "no",
        logging_strategy="epoch" if args.eval_each_epoch else "no",
        save_strategy="no",
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        report_to="none",
        seed=fold_seed,
        data_seed=fold_seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(),
        processing_class=tokenizer,
    )

    if dist.is_initialized():
        monitored_gpus = [torch.cuda.current_device()]
    else:
        monitored_gpus = list(range(torch.cuda.device_count()))

    for gpu_index in monitored_gpus:
        torch.cuda.reset_peak_memory_stats(gpu_index)

    monitor = ZeusMonitor(
        gpu_indices=monitored_gpus,
        cpu_indices=[],
    )
    window_name = f"fold_{fold}_training"
    monitor.begin_window(window_name)
    trainer.train()
    measurement = monitor.end_window(window_name)

    if trainer.is_world_process_zero() and args.eval_each_epoch:
        history_path = fold_output_dir / "epoch_history.json"
        history_path.write_text(
            json.dumps(trainer.state.log_history, indent=2),
            encoding="utf-8",
        )

    local_peak_memory_gb = max(
        torch.cuda.max_memory_allocated(gpu_index)
        for gpu_index in monitored_gpus
    ) / 1_000_000_000

    energy_j, time_s, memory_gb = aggregate_resources(
        measurement.total_energy,
        measurement.time,
        local_peak_memory_gb,
    )

    if args.convergence_pilot:
        fold_result = None
        if trainer.is_world_process_zero():
            fold_result = {
                "fold": fold,
                "seed": fold_seed,
                "train_gpu_energy_j": energy_j,
                "train_gpu_energy_kwh": energy_j / 3_600_000,
                "train_time_s": time_s,
                "peak_gpu_memory_gb": memory_gb,
            }

        if dist.is_initialized():
            dist.barrier()

        del trainer, model, datasets, monitor
        gc.collect()
        torch.cuda.empty_cache()
        return fold_result

    trainer.compute_metrics = None
    calibration_window_name = f"fold_{fold}_threshold_calibration"
    monitor.begin_window(calibration_window_name)
    calibration_output = trainer.predict(
        datasets["train"],
        metric_key_prefix="threshold_calibration",
    )
    thresholds = optimize_mcc_thresholds(
        sigmoid(calibration_output.predictions),
        calibration_output.label_ids,
    )
    calibration_measurement = monitor.end_window(
        calibration_window_name
    )
    (
        calibration_energy_j,
        calibration_time_s,
        _,
    ) = aggregate_resources(
        calibration_measurement.total_energy,
        calibration_measurement.time,
        0.0,
    )

    trainer.compute_metrics = build_compute_metrics(thresholds)
    model.config.deeploc_thresholds = thresholds.tolist()
    metrics = trainer.evaluate()
    metrics.update(
        {
            "train_gpu_energy_j": energy_j,
            "train_gpu_energy_kwh": energy_j / 3_600_000,
            "train_time_s": time_s,
            "threshold_calibration_gpu_energy_j":
                calibration_energy_j,
            "threshold_calibration_gpu_energy_kwh":
                calibration_energy_j / 3_600_000,
            "threshold_calibration_time_s": calibration_time_s,
            "pipeline_gpu_energy_j":
                energy_j + calibration_energy_j,
            "pipeline_gpu_energy_kwh":
                (energy_j + calibration_energy_j) / 3_600_000,
            "pipeline_time_s": time_s + calibration_time_s,
            "peak_gpu_memory_gb": memory_gb,
        }
    )

    trainer.save_model()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    fold_result = None
    if trainer.is_world_process_zero():
        fold_result = {
            "fold": fold,
            "seed": fold_seed,
            "thresholds": {
                label: float(threshold)
                for label, threshold in zip(
                    DEEPLOC_LABELS,
                    thresholds,
                )
            },
            "label_weights": {
                label: float(weight)
                for label, weight in zip(
                    DEEPLOC_LABELS,
                    label_weights,
                )
            },
        }
        fold_result.update({name: float(metrics[name]) for name in RESULT_NAMES})

    if dist.is_initialized():
        dist.barrier()

    del trainer, model, datasets, monitor
    gc.collect()
    torch.cuda.empty_cache()

    return fold_result


def summarize_results(fold_results):
    summary = {}
    for name in RESULT_NAMES:
        values = [result[name] for result in fold_results]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else None
            ),
        }
    return summary


def print_summary(summary):
    print("\nResumen de validación cruzada (media ± desviación típica)")
    for name, values in summary.items():
        if values["std"] is None:
            print(f"{name}: {values['mean']:.6f} (un solo fold)")
        else:
            print(
                f"{name}: {values['mean']:.6f} "
                f"± {values['std']:.6f}"
            )


def run(args, model_loader=load_model) -> None:
    if args.convergence_pilot and not args.eval_each_epoch:
        raise ValueError(
            "--convergence_pilot requires --eval_each_epoch"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Este script requiere al menos una GPU CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fold_results = []
    folds = args.folds if args.folds is not None else range(5)

    for fold in folds:
        result = train_fold(args, tokenizer, fold, model_loader)
        if result is not None:
            fold_results.append(result)

    if fold_results:
        if args.convergence_pilot:
            results = {
                "config": vars(args),
                "folds": fold_results,
            }
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / "cross_validation_results.json"
            results_path.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            print(
                "\nPiloto completado. Curvas y recursos guardados en: "
                f"{results_path}"
            )
            return

        summary = summarize_results(fold_results)
        results = {
            "config": vars(args),
            "folds": fold_results,
            "summary": summary,
        }

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "cross_validation_results.json"
        results_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print_summary(summary)
        print(f"\nResultados guardados en: {results_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
