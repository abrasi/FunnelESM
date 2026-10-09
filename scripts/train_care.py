#!/usr/bin/env python3
"""Train one CARE architecture; supports single-GPU and torchrun workers."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
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

from funnel_esm.care import CARE_LABELS, load_care
from funnel_esm.metrics import (
    apply_thresholds,
    inverse_frequency_label_weights,
    multilabel_metrics,
    optimize_mcc_thresholds,
    sigmoid,
)
from funnel_esm.modeling_classification import (
    MaskedAttentionEsmForSequenceClassification,
)
from funnel_esm.modeling_funnel_esm import (
    FunnelEsmForSequenceClassification,
    FunnelEsmPoolQueryForSequenceClassification,
)


ARCHITECTURES = (
    "baseline",
    "mean_layer_6",
    "mean_layers_4_8",
    "pool_query_layer_6",
    "pool_query_layers_4_8",
)
ROPE_KEY_MAPPING = {
    "esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq":
        "esm.rotary_embeddings.inv_freq",
}


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--train_csv", required=True)
    result.add_argument("--test_csv")
    result.add_argument("--cluster_metadata_csv")
    result.add_argument("--output_dir", required=True)
    result.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    result.add_argument("--model_name", default="facebook/esm2_t12_35M_UR50D")
    result.add_argument("--batch_size", type=int, default=8)
    result.add_argument("--epochs", type=int, default=15)
    result.add_argument("--learning_rate", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--validation_seed", type=int, default=0)
    result.add_argument("--validation_size", type=float, default=0.1)
    result.add_argument("--max_residues", type=int, default=1022)
    result.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    result.add_argument("--focal_gamma", type=float, default=1.0)
    result.add_argument("--convergence_pilot", action="store_true")
    return result


def package_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def build_model(args):
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(CARE_LABELS)
    config.problem_type = "multi_label_classification"
    config.id2label = dict(enumerate(CARE_LABELS))
    config.label2id = {name: index for index, name in enumerate(CARE_LABELS)}
    config.deeploc_focal_gamma = args.focal_gamma

    if args.architecture == "baseline":
        model_class = MaskedAttentionEsmForSequenceClassification
    else:
        if args.architecture.startswith("mean_"):
            model_class = FunnelEsmForSequenceClassification
            config.funnel_pooling_type = "pairwise_masked_mean"
        else:
            model_class = FunnelEsmPoolQueryForSequenceClassification
            config.funnel_pooling_type = "pool_query_only_mean"
        config.funnel_pool_after_layers = (
            [4, 8] if "layers_4_8" in args.architecture else [6]
        )
        config.funnel_pool_stride = 2

    model = model_class.from_pretrained(
        args.model_name,
        config=config,
        key_mapping=ROPE_KEY_MAPPING,
    )
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    return model


def model_input_dataset(dataset):
    if "protein_id" in dataset.column_names:
        return dataset.remove_columns("protein_id")
    return dataset


def compute_care_metrics(eval_prediction, thresholds=None, force_at_least_one=True, inclusive=True):
    if thresholds is None:
        thresholds = np.full(len(CARE_LABELS), 0.5, dtype=np.float64)
    probabilities = sigmoid(eval_prediction.predictions)
    labels = np.asarray(eval_prediction.label_ids, dtype=np.int64)
    metrics = multilabel_metrics(
        labels,
        probabilities,
        thresholds,
        force_at_least_one=force_at_least_one,
        inclusive=inclusive,
        label_names=CARE_LABELS,
    )
    predicted_ec1 = np.argmax(probabilities, axis=1)
    row_indices = np.arange(len(labels))
    metrics["top1_ec1_accuracy"] = float(
        labels[row_indices, predicted_ec1].mean()
    )
    return metrics


def aggregate_resources(energy_j, time_s, memory_gb):
    """Sum GPU energy and take max time/memory across torchrun workers."""
    if not dist.is_initialized():
        return energy_j, time_s, memory_gb
    device = torch.device("cuda", torch.cuda.current_device())
    energy = torch.tensor(energy_j, dtype=torch.float64, device=device)
    duration = torch.tensor(time_s, dtype=torch.float64, device=device)
    memory = torch.tensor(memory_gb, dtype=torch.float64, device=device)
    dist.all_reduce(energy, op=dist.ReduceOp.SUM)
    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    return energy.item(), duration.item(), memory.item()


def save_test_predictions(path, protein_ids, probabilities, labels, predictions):
    import csv

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    expected_shape = (len(protein_ids), len(CARE_LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError("Test probabilities do not match protein identifiers")
    if labels.shape != expected_shape or predictions.shape != expected_shape:
        raise ValueError("Test labels/predictions have an unexpected shape")

    header = ["protein_id"]
    header.extend(f"prob_{label}" for label in CARE_LABELS)
    header.extend(f"true_{label}" for label in CARE_LABELS)
    header.extend(f"pred_{label}" for label in CARE_LABELS)
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row, protein_id in enumerate(protein_ids):
            writer.writerow(
                [
                    protein_id,
                    *probabilities[row].tolist(),
                    *labels[row].tolist(),
                    *predictions[row].tolist(),
                ]
            )


def main():
    args = parser().parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, learning_rate, and batch_size must be positive")
    if args.focal_gamma < 0:
        raise ValueError("focal_gamma cannot be negative")
    if args.convergence_pilot and args.test_csv:
        raise ValueError("Do not pass --test_csv to a convergence pilot")
    if not args.convergence_pilot and not args.test_csv:
        raise ValueError("--test_csv is required for final experiments")
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
        test_csv=None if args.convergence_pilot else args.test_csv,
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
        max_residues=args.max_residues,
        cluster_metadata_csv=args.cluster_metadata_csv,
    )

    label_weights = inverse_frequency_label_weights(data["train"]["labels"])
    model = build_model(args)
    model.set_label_weights(torch.tensor(label_weights, dtype=torch.float32))

    final_run = not args.convergence_pilot
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch" if final_run else "no",
        load_best_model_at_end=final_run,
        metric_for_best_model="eval_loss" if final_run else None,
        greater_is_better=False if final_run else None,
        save_total_limit=1 if final_run else None,
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=model_input_dataset(data["train"]),
        eval_dataset=model_input_dataset(data["validation"]),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda prediction: compute_care_metrics(
            prediction,
            force_at_least_one=not final_run,
            inclusive=not final_run,
        ),
        processing_class=tokenizer,
    )
    is_main_process = trainer.is_world_process_zero()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        monitored_gpus = [torch.cuda.current_device()]
    else:
        monitored_gpus = list(range(torch.cuda.device_count()))
    gpu_names = [torch.cuda.get_device_name(index) for index in monitored_gpus]

    if is_main_process:
        training_protocol = {
            "phase": "final" if final_run else "convergence_pilot",
            "training_arguments": training_args.to_dict(),
            "effective_train_batch_size": args.batch_size * world_size,
            "data_protocol": {
                "train_csv": str(Path(args.train_csv).resolve()),
                "test_csv": (
                    str(Path(args.test_csv).resolve()) if final_run else None
                ),
                "test_loaded_during_pilot": False,
                "validation_size": args.validation_size,
                "validation_seed": args.validation_seed,
                "validation_grouping": (
                    "clusterRes50"
                    if args.cluster_metadata_csv
                    else "exact_sequence"
                ),
                "checkpoint_selection": "minimum eval_loss",
                "threshold_calibration": (
                    "all official training examples" if final_run else None
                ),
                "test_used_for_selection": False,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "accelerate": package_version("accelerate"),
                "datasets": package_version("datasets"),
                "zeus": package_version("zeus", "zeus-ml"),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": gpu_names,
                "world_size": world_size,
            },
        }
        (output_dir / "training_protocol.json").write_text(
            json.dumps(training_protocol, indent=2), encoding="utf-8"
        )

    for index in monitored_gpus:
        torch.cuda.reset_peak_memory_stats(index)
    monitor = ZeusMonitor(gpu_indices=monitored_gpus, cpu_indices=[])
    monitor.begin_window("care_training")
    trainer.train()
    local_training_measurement = monitor.end_window("care_training")
    local_peak_memory_gb = max(
        torch.cuda.max_memory_allocated(index) for index in monitored_gpus
    ) / 1_000_000_000
    train_energy_j, train_time_s, peak_memory_gb = aggregate_resources(
        local_training_measurement.total_energy,
        local_training_measurement.time,
        local_peak_memory_gb,
    )

    if trainer.state.best_model_checkpoint is None and final_run:
        raise RuntimeError("Final training did not select a best checkpoint")
    evaluation_records = [
        record for record in trainer.state.log_history if "eval_loss" in record
    ]
    if not evaluation_records:
        raise RuntimeError("No epoch-level validation results were recorded")
    best_record = min(evaluation_records, key=lambda record: record["eval_loss"])

    if is_main_process:
        (output_dir / "epoch_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
        )

    train_resources = {
        "train_gpu_energy_j": float(train_energy_j),
        "train_gpu_energy_kwh": float(train_energy_j / 3_600_000),
        "train_time_s": float(train_time_s),
        "peak_gpu_memory_gb": float(peak_memory_gb),
    }

    if not final_run:
        pilot_result = {
            "phase": "pilot",
            "architecture": args.architecture,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "best_epoch": int(round(best_record["epoch"])),
            "best_validation_loss": float(best_record["eval_loss"]),
            "best_validation_metrics": {
                key: float(value)
                for key, value in best_record.items()
                if key.startswith("eval_")
                and isinstance(value, (int, float, np.number))
            },
            "resources": train_resources,
        }
        if is_main_process:
            (output_dir / "run_result.json").write_text(
                json.dumps(pilot_result, indent=2), encoding="utf-8"
            )
            print(json.dumps(pilot_result, indent=2), flush=True)
        if dist.is_initialized():
            dist.barrier()
        del trainer, model, data, monitor
        gc.collect()
        torch.cuda.empty_cache()
        return

    trainer.compute_metrics = None
    monitor.begin_window("care_threshold_calibration")
    calibration_output = trainer.predict(
        model_input_dataset(data["threshold_calibration"]),
        metric_key_prefix="threshold_calibration",
    )
    thresholds = optimize_mcc_thresholds(
        sigmoid(calibration_output.predictions),
        calibration_output.label_ids,
    )
    local_calibration = monitor.end_window("care_threshold_calibration")
    calibration_energy_j, calibration_time_s, _ = aggregate_resources(
        local_calibration.total_energy,
        local_calibration.time,
        0.0,
    )

    trainer.compute_metrics = lambda prediction: compute_care_metrics(
        prediction,
        thresholds=thresholds,
        force_at_least_one=False,
        inclusive=False,
    )
    model.config.deeploc_thresholds = thresholds.tolist()
    model.config.deeploc_force_at_least_one_label = False
    model.config.deeploc_threshold_comparison = ">"

    monitor.begin_window("care_official_test")
    test_output = trainer.predict(
        model_input_dataset(data["test"]),
        metric_key_prefix="test",
    )
    local_test_measurement = monitor.end_window("care_official_test")
    test_energy_j, test_time_s, _ = aggregate_resources(
        local_test_measurement.total_energy,
        local_test_measurement.time,
        0.0,
    )
    probabilities = sigmoid(test_output.predictions)
    predictions = apply_thresholds(
        probabilities,
        thresholds,
        force_at_least_one=False,
        inclusive=False,
    )
    test_metrics = {
        key.removeprefix("test_"): float(value)
        for key, value in test_output.metrics.items()
        if key.startswith("test_")
        and isinstance(value, (float, int, np.number))
    }
    pipeline_energy_j = train_energy_j + calibration_energy_j
    pipeline_time_s = train_time_s + calibration_time_s
    final_resources = {
        **train_resources,
        "threshold_calibration_gpu_energy_j": float(calibration_energy_j),
        "threshold_calibration_gpu_energy_kwh": float(
            calibration_energy_j / 3_600_000
        ),
        "threshold_calibration_time_s": float(calibration_time_s),
        "pipeline_gpu_energy_j": float(pipeline_energy_j),
        "pipeline_gpu_energy_kwh": float(pipeline_energy_j / 3_600_000),
        "pipeline_time_s": float(pipeline_time_s),
        "test_gpu_energy_j": float(test_energy_j),
        "test_gpu_energy_kwh": float(test_energy_j / 3_600_000),
        "test_time_s": float(test_time_s),
    }

    if is_main_process:
        trainer.save_model(str(output_dir / "best_model"))
        tokenizer.save_pretrained(output_dir / "best_model")
        save_test_predictions(
            output_dir / "test_predictions.csv",
            data["test"]["protein_id"],
            probabilities,
            test_output.label_ids,
            predictions,
        )
        result = {
            "phase": "final",
            "architecture": args.architecture,
            "seed": args.seed,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "model_selection": {
                "metric": "eval_loss",
                "best_validation_loss": float(trainer.state.best_metric),
                "best_epoch": float(best_record["epoch"]),
                "maximum_epochs": args.epochs,
                "early_stopping": False,
            },
            "threshold_calibration": {
                "metric": "per_label_mcc",
                "data": "all official training examples",
                "test_labels_used": False,
                "comparison": "probability > threshold",
            },
            "thresholds": {
                label: float(threshold)
                for label, threshold in zip(CARE_LABELS, thresholds)
            },
            "label_weights": {
                label: float(weight)
                for label, weight in zip(CARE_LABELS, label_weights)
            },
            "training_examples": len(data["train"]),
            "validation_examples": len(data["validation"]),
            "test_examples": len(data["test"]),
            "test_metrics": test_metrics,
            "resources": final_resources,
        }
        (output_dir / "run_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (output_dir / "care_test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
    del trainer, model, data, monitor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
