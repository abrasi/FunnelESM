#!/usr/bin/env python3
"""Train and evaluate the frozen ESM-2 baseline with 5-fold cross-validation."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import platform
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

from funnel_esm.deeploc import DEEPLOC_LABELS, load_deeploc
from funnel_esm.metrics import (
    MCC_METRIC_NAMES,
    apply_thresholds,
    build_compute_metrics,
    inverse_frequency_label_weights,
    label_metric_name,
    optimize_mcc_thresholds,
    sigmoid,
)
from funnel_esm.modeling_classification import (
    MaskedAttentionEsmForSequenceClassification,
)


def package_version(*distribution_names):
    """Return the installed version for the first matching distribution."""
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate fixed from the fold-0 convergence pilot.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=1.0,
        help="Focal-loss gamma. DeepLoc 2.0 uses 1.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
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
            "Compatibility flag used by the convergence pilot. Final "
            "training always evaluates and records every epoch."
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
    parser.add_argument(
        "--skip_existing_folds",
        action="store_true",
        help=(
            "Reuse completed fold_result.json files. Intended for "
            "recovering an interrupted final experiment."
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


def save_fold_predictions(
    output_path,
    protein_ids,
    probabilities,
    labels,
    predictions,
):
    """Save fold-level outputs required for paired statistical analyses."""
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    expected_shape = (len(protein_ids), len(DEEPLOC_LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError("probabilities do not match protein identifiers")
    if labels.shape != expected_shape or predictions.shape != expected_shape:
        raise ValueError("labels and predictions must match probabilities")

    label_names = [label_metric_name(label) for label in DEEPLOC_LABELS]
    fieldnames = ["protein_id"]
    fieldnames.extend(f"prob_{name}" for name in label_names)
    fieldnames.extend(f"true_{name}" for name in label_names)
    fieldnames.extend(f"pred_{name}" for name in label_names)

    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, protein_id in enumerate(protein_ids):
            row = {"protein_id": protein_id}
            row.update(
                {
                    f"prob_{name}": float(probabilities[row_index, index])
                    for index, name in enumerate(label_names)
                }
            )
            row.update(
                {
                    f"true_{name}": int(labels[row_index, index])
                    for index, name in enumerate(label_names)
                }
            )
            row.update(
                {
                    f"pred_{name}": int(predictions[row_index, index])
                    for index, name in enumerate(label_names)
                }
            )
            writer.writerow(row)


def model_input_dataset(dataset):
    """Remove analysis-only identifiers before Hugging Face collation."""
    if "protein_id" in dataset.column_names:
        return dataset.remove_columns("protein_id")
    return dataset


def train_fold(args, tokenizer, fold, model_loader=load_model):
    """Train one fold and return its metrics on the main process."""
    fold_seed = args.seed + fold
    set_seed(fold_seed)
    fold_output_dir = Path(args.output_dir) / f"fold_{fold}"
    final_experiment = not args.convergence_pilot

    datasets = load_deeploc(
        args.dataset_path,
        tokenizer,
        validation_partition=fold,
        use_internal_validation=final_experiment,
        internal_validation_size=2048,
        internal_validation_seed=0,
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
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch" if final_experiment else "no",
        load_best_model_at_end=final_experiment,
        metric_for_best_model="eval_loss" if final_experiment else None,
        greater_is_better=False if final_experiment else None,
        save_total_limit=1 if final_experiment else None,
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        report_to="none",
        seed=fold_seed,
        data_seed=fold_seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=model_input_dataset(datasets["train"]),
        eval_dataset=model_input_dataset(datasets["validation"]),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(
            force_at_least_one=not final_experiment,
            inclusive=not final_experiment,
        ),
        processing_class=tokenizer,
    )

    if trainer.is_world_process_zero():
        world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else 1
        )
        protocol = {
            "training_arguments": training_args.to_dict(),
            "effective_train_batch_size": (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * world_size
            ),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "accelerate": package_version("accelerate"),
                "datasets": package_version("datasets"),
                "zeus": package_version("zeus", "zeus-ml"),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(
                    torch.cuda.current_device()
                ),
                "world_size": world_size,
            },
        }
        if final_experiment:
            protocol["data_protocol"] = {
                "outer_evaluation_partition": fold,
                "internal_validation_size": 2048,
                "internal_validation_seed": 0,
                "checkpoint_selection_data": "internal_validation",
                "threshold_calibration_data": (
                    "four_official_outer_training_partitions"
                ),
                "outer_evaluation_labels_used_before_evaluation": False,
            }
        protocol_path = fold_output_dir / "training_protocol.json"
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(
            json.dumps(protocol, indent=2),
            encoding="utf-8",
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

    if trainer.is_world_process_zero():
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

    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError(
            "Final training finished without selecting a best checkpoint."
        )

    evaluation_records = [
        record
        for record in trainer.state.log_history
        if "eval_loss" in record
    ]
    if not evaluation_records:
        raise RuntimeError("No epoch-level validation losses were recorded.")
    best_record = min(
        evaluation_records,
        key=lambda record: record["eval_loss"],
    )

    trainer.compute_metrics = None
    calibration_window_name = f"fold_{fold}_threshold_calibration"
    monitor.begin_window(calibration_window_name)
    calibration_output = trainer.predict(
        model_input_dataset(datasets["threshold_calibration"]),
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

    trainer.compute_metrics = build_compute_metrics(
        thresholds,
        force_at_least_one=False,
        inclusive=False,
    )
    model.config.deeploc_thresholds = thresholds.tolist()
    model.config.deeploc_force_at_least_one_label = False
    model.config.deeploc_threshold_comparison = ">"
    evaluation_output = trainer.predict(
        model_input_dataset(datasets["outer_validation"]),
        metric_key_prefix="eval",
    )
    metrics = evaluation_output.metrics
    evaluation_probabilities = sigmoid(evaluation_output.predictions)
    evaluation_predictions = apply_thresholds(
        evaluation_probabilities,
        thresholds,
        force_at_least_one=False,
        inclusive=False,
    )
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
            "model_selection": {
                "metric": "eval_loss",
                "data": (
                    "fixed_2048_internal_validation_examples_from_the_"
                    "four_outer_training_partitions"
                ),
                "internal_validation_seed": 0,
                "best_metric": float(trainer.state.best_metric),
                "best_epoch": float(best_record["epoch"]),
                "best_model_checkpoint": (
                    trainer.state.best_model_checkpoint
                ),
                "maximum_epochs": args.epochs,
                "early_stopping": False,
            },
            "threshold_calibration": {
                "metric": "per_label_mcc",
                "data": "four_official_training_partitions",
                "evaluated_partition_labels_used": False,
                "force_at_least_one_label": False,
                "threshold_comparison": "probability > threshold",
            },
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
        save_fold_predictions(
            fold_output_dir / "outer_fold_predictions.csv",
            datasets["outer_validation"]["protein_id"],
            evaluation_probabilities,
            evaluation_output.label_ids,
            evaluation_predictions,
        )
        fold_result_path = fold_output_dir / "fold_result.json"
        fold_result_path.write_text(
            json.dumps(fold_result, indent=2),
            encoding="utf-8",
        )

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
                float(np.std(values, ddof=0))
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
    if (
        args.mixed_precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "La GPU no soporta BF16. Usa --mixed_precision no para "
            "ejecutar en FP32; no cambies automáticamente a FP16."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fold_results = []
    folds = args.folds if args.folds is not None else range(5)

    for fold in folds:
        fold_result_path = (
            Path(args.output_dir) / f"fold_{fold}" / "fold_result.json"
        )
        if (
            args.skip_existing_folds
            and not args.convergence_pilot
            and fold_result_path.exists()
        ):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Reusing completed fold {fold}: {fold_result_path}")
                result = json.loads(
                    fold_result_path.read_text(encoding="utf-8")
                )
            else:
                result = None
            if dist.is_initialized():
                dist.barrier()
        else:
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
