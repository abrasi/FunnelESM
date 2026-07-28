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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from zeus.monitor import ZeusMonitor

from funnel_esm.data import DEEPLOC_LABELS, load_deeploc
from funnel_esm.metrics import compute_metrics


RESULT_NAMES = (
    "eval_loss",
    "eval_macro_f1",
    "eval_micro_f1",
    "eval_jaccard",
    "eval_accuracy",
    "eval_num_labels_predicted",
    "train_gpu_energy_j",
    "train_gpu_energy_kwh",
    "train_time_s",
    "peak_gpu_memory_gb",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        default="facebook/esm2_t12_35M_UR50D",
    )
    return parser


def parse_args():
    return build_parser().parse_args()


def load_model(args):
    return AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(DEEPLOC_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(DEEPLOC_LABELS)),
        label2id={label: index for index, label in enumerate(DEEPLOC_LABELS)},
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
    set_seed(args.seed)
    fold_output_dir = Path(args.output_dir) / f"fold_{fold}"

    datasets = load_deeploc(
        args.dataset_path,
        tokenizer,
        validation_partition=fold,
    )

    model = model_loader(args)

    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
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

    local_peak_memory_gb = max(
        torch.cuda.max_memory_allocated(gpu_index)
        for gpu_index in monitored_gpus
    ) / 1_000_000_000

    energy_j, time_s, memory_gb = aggregate_resources(
        measurement.total_energy,
        measurement.time,
        local_peak_memory_gb,
    )

    metrics = trainer.evaluate()
    metrics.update(
        {
            "train_gpu_energy_j": energy_j,
            "train_gpu_energy_kwh": energy_j / 3_600_000,
            "train_time_s": time_s,
            "peak_gpu_memory_gb": memory_gb,
        }
    )

    trainer.save_model()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    fold_result = None
    if trainer.is_world_process_zero():
        fold_result = {"fold": fold}
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
            "std": float(np.std(values, ddof=1)),
        }
    return summary


def print_summary(summary):
    print("\nResumen de validación cruzada (media ± desviación típica)")
    for name, values in summary.items():
        print(f"{name}: {values['mean']:.6f} ± {values['std']:.6f}")


def run(args, model_loader=load_model) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Este script requiere al menos una GPU CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fold_results = []

    for fold in range(5):
        result = train_fold(args, tokenizer, fold, model_loader)
        if result is not None:
            fold_results.append(result)

    if fold_results:
        summary = summarize_results(fold_results)
        results = {
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