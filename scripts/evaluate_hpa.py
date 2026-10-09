#!/usr/bin/env python3
"""Evaluate the completed five-fold models on the independent HPA set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from zeus.monitor import ZeusMonitor

from funnel_esm.deeploc import DEEPLOC_LABELS, HPA_LABELS, load_hpa
from funnel_esm.metrics import (
    apply_thresholds,
    label_metric_name,
    multilabel_metrics,
    sigmoid,
)
from funnel_esm.modeling_classification import (
    MaskedAttentionEsmForSequenceClassification,
)
from funnel_esm.modeling_funnel_esm import (
    FunnelEsmForSequenceClassification,
    FunnelEsmPoolQueryForSequenceClassification,
)


ARCHITECTURE_MODEL_CLASSES = {
    "baseline": MaskedAttentionEsmForSequenceClassification,
    "mean_layer_6": FunnelEsmForSequenceClassification,
    "mean_layers_4_8": FunnelEsmForSequenceClassification,
    "pool_query_layer_6": FunnelEsmPoolQueryForSequenceClassification,
    "pool_query_layers_4_8": FunnelEsmPoolQueryForSequenceClassification,
}

ROPE_KEY_MAPPING = {
    "esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq":
        "esm.rotary_embeddings.inv_freq",
}

# DeepLoc 2.0 Table 2 evaluates six HPA locations. The downloaded HPA file
# also contains Lysosome/Vacuole and Peroxisome, but with only 2 and 7
# positive examples, respectively; they are excluded from the benchmark.
HPA_BENCHMARK_LABELS = tuple(
    label
    for label in HPA_LABELS
    if label not in {"Lysosome/Vacuole", "Peroxisome"}
)
HPA_OUTPUT_INDICES = tuple(
    DEEPLOC_LABELS.index(label) for label in HPA_BENCHMARK_LABELS
)
HPA_DATASET_INDICES = tuple(
    HPA_LABELS.index(label) for label in HPA_BENCHMARK_LABELS
)

METRIC_NAMES = (
    "macro_f1",
    "micro_f1",
    "jaccard",
    "accuracy",
    "num_labels_predicted",
    "macro_mcc",
    *tuple(
        f"mcc_{label_metric_name(label)}"
        for label in HPA_BENCHMARK_LABELS
    ),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every completed fold and its five-model ensemble "
            "on the independent DeepLoc 2.0 HPA test set."
        )
    )
    parser.add_argument(
        "--final_output_dir",
        required=True,
        help=(
            "Directory produced by run_final_experiments.py, containing "
            "one subdirectory per architecture."
        ),
    )
    parser.add_argument("--hpa_path", required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Defaults to <final_output_dir>/hpa_evaluation. Use a new "
            "directory if the HPA protocol changes."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Must match the final experiment manifest. If omitted, it is "
            "read from that manifest."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default=None,
        help=(
            "Must match the final experiment manifest. If omitted, it is "
            "read from that manifest."
        ),
    )
    return parser


def inference_dataset(dataset):
    analysis_columns = [
        column
        for column in ("protein_id", "hpa_labels")
        if column in dataset.column_names
    ]
    return dataset.remove_columns(analysis_columns)


def load_thresholds(fold_result_path):
    fold_result = json.loads(
        fold_result_path.read_text(encoding="utf-8")
    )
    threshold_dict = fold_result.get("thresholds")
    if threshold_dict is None:
        raise RuntimeError(
            f"No calibrated thresholds in {fold_result_path}"
        )
    try:
        return np.asarray(
            [threshold_dict[label] for label in DEEPLOC_LABELS],
            dtype=np.float64,
        )
    except KeyError as error:
        raise RuntimeError(
            f"Missing threshold {error.args[0]!r} in {fold_result_path}"
        ) from error


def save_predictions(
    output_path,
    protein_ids,
    probabilities,
    labels,
    predictions,
):
    label_names = [
        label_metric_name(label) for label in HPA_BENCHMARK_LABELS
    ]
    fieldnames = ["protein_id"]
    fieldnames.extend(f"prob_{name}" for name in label_names)
    fieldnames.extend(f"true_{name}" for name in label_names)
    fieldnames.extend(f"pred_{name}" for name in label_names)

    with Path(output_path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, protein_id in enumerate(protein_ids):
            row = {"protein_id": protein_id}
            for label_index, name in enumerate(label_names):
                row[f"prob_{name}"] = float(
                    probabilities[row_index, label_index]
                )
                row[f"true_{name}"] = int(
                    labels[row_index, label_index]
                )
                row[f"pred_{name}"] = int(
                    predictions[row_index, label_index]
                )
            writer.writerow(row)


def compute_hpa_metrics(labels, probabilities, thresholds):
    metrics = multilabel_metrics(
        labels,
        probabilities,
        thresholds,
        force_at_least_one=False,
        inclusive=False,
        label_names=HPA_BENCHMARK_LABELS,
    )
    metrics["num_labels_true"] = float(labels.sum(axis=1).mean())
    return {name: float(value) for name, value in metrics.items()}


def summarize_fold_results(fold_results):
    summary = {}
    for name in (
        *METRIC_NAMES,
        "inference_gpu_energy_j",
        "inference_gpu_energy_kwh",
        "inference_time_s",
        "peak_gpu_memory_gb",
    ):
        values = np.asarray(
            [fold_result[name] for fold_result in fold_results],
            dtype=np.float64,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def evaluate_architecture(
    architecture,
    architecture_dir,
    output_dir,
    hpa_dataset,
    protein_ids,
    labels,
    tokenizer,
    args,
):
    model_class = ARCHITECTURE_MODEL_CLASSES[architecture]
    fold_results = []
    fold_probabilities = []
    fold_thresholds = []

    for fold in range(5):
        model_dir = architecture_dir / f"fold_{fold}"
        fold_result_path = model_dir / "fold_result.json"
        if not fold_result_path.exists():
            raise FileNotFoundError(
                f"Missing completed fold: {fold_result_path}"
            )

        thresholds_10 = load_thresholds(fold_result_path)
        thresholds = thresholds_10[list(HPA_OUTPUT_INDICES)]
        model = model_class.from_pretrained(
            model_dir,
            key_mapping=ROPE_KEY_MAPPING,
        )
        config_thresholds = getattr(
            model.config,
            "deeploc_thresholds",
            None,
        )
        if config_thresholds is not None and not np.allclose(
            thresholds_10,
            np.asarray(config_thresholds, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Threshold mismatch between config and {fold_result_path}"
            )

        fold_output_dir = output_dir / f"fold_{fold}"
        prediction_args = TrainingArguments(
            output_dir=str(fold_output_dir / "trainer"),
            per_device_eval_batch_size=args.batch_size,
            fp16=args.mixed_precision == "fp16",
            bf16=args.mixed_precision == "bf16",
            report_to="none",
        )
        if prediction_args.n_gpu != 1:
            raise RuntimeError(
                "HPA evaluation must use exactly one GPU. Launch it with "
                "torchrun --standalone --nproc_per_node=1, as in the "
                "training experiments."
            )
        trainer = Trainer(
            model=model,
            args=prediction_args,
            data_collator=DataCollatorWithPadding(tokenizer),
            processing_class=tokenizer,
        )

        gpu_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(gpu_index)
        monitor = ZeusMonitor(gpu_indices=[gpu_index], cpu_indices=[])
        window_name = f"hpa_{architecture}_fold_{fold}"
        monitor.begin_window(window_name)
        prediction_output = trainer.predict(
            inference_dataset(hpa_dataset),
            metric_key_prefix="hpa",
        )
        measurement = monitor.end_window(window_name)

        logits = prediction_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities_10 = sigmoid(logits)
        probabilities = probabilities_10[:, list(HPA_OUTPUT_INDICES)]
        predictions = apply_thresholds(
            probabilities,
            thresholds,
            force_at_least_one=False,
            inclusive=False,
        )
        metrics = compute_hpa_metrics(
            labels,
            probabilities,
            thresholds,
        )
        metrics.update(
            {
                "fold": fold,
                "inference_gpu_energy_j": float(
                    measurement.total_energy
                ),
                "inference_gpu_energy_kwh": float(
                    measurement.total_energy / 3_600_000
                ),
                "inference_time_s": float(measurement.time),
                "peak_gpu_memory_gb": float(
                    torch.cuda.max_memory_allocated(gpu_index)
                    / 1_000_000_000
                ),
                "gpu_index": gpu_index,
                "gpu_name": torch.cuda.get_device_name(gpu_index),
                "trainer_n_gpu": prediction_args.n_gpu,
                "thresholds": {
                    label: float(threshold)
                    for label, threshold in zip(
                        HPA_BENCHMARK_LABELS,
                        thresholds,
                    )
                },
            }
        )

        fold_output_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(
            fold_output_dir / "hpa_predictions.csv",
            protein_ids,
            probabilities,
            labels,
            predictions,
        )
        (fold_output_dir / "hpa_result.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        fold_results.append(metrics)
        fold_probabilities.append(probabilities)
        fold_thresholds.append(thresholds)

        del trainer, model, monitor, prediction_output
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_probabilities = np.mean(fold_probabilities, axis=0)
    ensemble_thresholds = np.mean(fold_thresholds, axis=0)
    ensemble_predictions = apply_thresholds(
        ensemble_probabilities,
        ensemble_thresholds,
        force_at_least_one=False,
        inclusive=False,
    )
    ensemble_metrics = compute_hpa_metrics(
        labels,
        ensemble_probabilities,
        ensemble_thresholds,
    )
    ensemble_metrics.update(
        {
            "thresholds": {
                label: float(threshold)
                for label, threshold in zip(
                    HPA_BENCHMARK_LABELS,
                    ensemble_thresholds,
                )
            },
            "inference_gpu_energy_j": float(
                sum(
                    result["inference_gpu_energy_j"]
                    for result in fold_results
                )
            ),
            "inference_gpu_energy_kwh": float(
                sum(
                    result["inference_gpu_energy_kwh"]
                    for result in fold_results
                )
            ),
            "inference_time_s": float(
                sum(result["inference_time_s"] for result in fold_results)
            ),
            "peak_gpu_memory_gb": float(
                max(result["peak_gpu_memory_gb"] for result in fold_results)
            ),
        }
    )
    save_predictions(
        output_dir / "hpa_ensemble_predictions.csv",
        protein_ids,
        ensemble_probabilities,
        labels,
        ensemble_predictions,
    )

    architecture_results = {
        "architecture": architecture,
        "evaluated_labels": list(HPA_BENCHMARK_LABELS),
        "unannotated_classifier_outputs_excluded": [
            label for label in DEEPLOC_LABELS if label not in HPA_LABELS
        ],
        "low_support_hpa_labels_excluded": [
            "Lysosome/Vacuole",
            "Peroxisome",
        ],
        "label_support": {
            label: int(labels[:, index].sum())
            for index, label in enumerate(HPA_BENCHMARK_LABELS)
        },
        "folds": fold_results,
        "fold_summary": summarize_fold_results(fold_results),
        "ensemble": ensemble_metrics,
        "ensemble_definition": {
            "probabilities": "arithmetic_mean_across_five_fold_models",
            "thresholds": "arithmetic_mean_across_five_fold_thresholds",
            "comparison": "mean_probability > mean_threshold",
            "force_at_least_one_label": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hpa_results.json").write_text(
        json.dumps(architecture_results, indent=2),
        encoding="utf-8",
    )
    return architecture_results


def main():
    args = build_parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_with_torchrun = "LOCAL_RANK" in os.environ
    if not launched_with_torchrun or world_size != 1:
        raise RuntimeError(
            "evaluate_hpa.py is the single-GPU worker. Launch it through "
            "run_hpa_evaluation.py so it uses exactly one torchrun "
            "process, like the final training experiments."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("HPA evaluation requires a CUDA GPU.")

    final_output_dir = Path(args.final_output_dir).resolve()
    final_manifest_path = (
        final_output_dir / "final_experiment_manifest.json"
    )
    if not final_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing final experiment manifest: {final_manifest_path}"
        )
    final_manifest = json.loads(
        final_manifest_path.read_text(encoding="utf-8")
    )
    if final_manifest.get("protocol_version") != 2:
        raise RuntimeError(
            "The final results were not produced by protocol version 2."
        )
    if final_manifest.get("nproc_per_node") != 1:
        raise RuntimeError(
            "HPA single-GPU resources are only comparable with final "
            "experiments produced using nproc_per_node=1."
        )

    final_batch_size = int(final_manifest["batch_size_per_device"])
    if args.batch_size is not None and args.batch_size != final_batch_size:
        raise RuntimeError(
            "--batch_size must match the final experiment manifest: "
            f"{final_batch_size}."
        )
    args.batch_size = final_batch_size

    final_mixed_precision = final_manifest["mixed_precision"]
    if (
        args.mixed_precision is not None
        and args.mixed_precision != final_mixed_precision
    ):
        raise RuntimeError(
            "--mixed_precision must match the final experiment manifest: "
            f"{final_mixed_precision}."
        )
    args.mixed_precision = final_mixed_precision
    if (
        args.mixed_precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "The final protocol uses BF16, but this GPU does not support "
            "it. Use the same compatible GPU as for training."
        )

    model_name = final_manifest["model_name"]
    manifest_architectures = [
        run["name"] for run in final_manifest.get("runs", [])
    ]
    if manifest_architectures != list(ARCHITECTURE_MODEL_CLASSES):
        raise RuntimeError(
            "The final manifest does not contain the five expected "
            "architectures in their fixed order."
        )
    architectures = manifest_architectures

    training_gpu_names = set()
    for architecture in architectures:
        for fold in range(5):
            fold_dir = (
                final_output_dir
                / architecture
                / f"fold_{fold}"
            )
            fold_result_path = fold_dir / "fold_result.json"
            if not fold_result_path.exists():
                raise FileNotFoundError(
                    f"Missing completed fold: {fold_result_path}"
                )
            fold_result = json.loads(
                fold_result_path.read_text(encoding="utf-8")
            )
            if fold_result.get("fold") != fold:
                raise RuntimeError(
                    f"Fold identity mismatch in {fold_result_path}"
                )

            training_protocol_path = (
                fold_dir / "training_protocol.json"
            )
            if not training_protocol_path.exists():
                raise FileNotFoundError(
                    "Missing training protocol: "
                    f"{training_protocol_path}"
                )
            training_protocol = json.loads(
                training_protocol_path.read_text(encoding="utf-8")
            )
            training_arguments = training_protocol[
                "training_arguments"
            ]
            if (
                training_arguments["per_device_eval_batch_size"]
                != args.batch_size
                or bool(training_arguments["fp16"])
                != (args.mixed_precision == "fp16")
                or bool(training_arguments["bf16"])
                != (args.mixed_precision == "bf16")
            ):
                raise RuntimeError(
                    "Fold evaluation settings do not match the final "
                    f"manifest: {training_protocol_path}"
                )
            training_gpu_names.add(
                training_protocol["runtime"]["gpu"]
            )

    if len(training_gpu_names) != 1:
        raise RuntimeError(
            "The final folds were not run on one consistent GPU model."
        )
    current_gpu_name = torch.cuda.get_device_name(
        torch.cuda.current_device()
    )
    if current_gpu_name not in training_gpu_names:
        raise RuntimeError(
            "HPA must use the same GPU model as the final experiments. "
            f"Training used {next(iter(training_gpu_names))!r}, but HPA "
            f"would use {current_gpu_name!r}."
        )

    hpa_path = Path(args.hpa_path).resolve()
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else final_output_dir / "hpa_evaluation"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hpa_dataset = load_hpa(hpa_path, tokenizer)
    protein_ids = list(hpa_dataset["protein_id"])
    all_hpa_labels = np.asarray(
        hpa_dataset["hpa_labels"],
        dtype=np.int64,
    )
    labels = all_hpa_labels[:, list(HPA_DATASET_INDICES)]
    hpa_label_support = {
        label: int(all_hpa_labels[:, index].sum())
        for index, label in enumerate(HPA_LABELS)
    }

    manifest = {
        "purpose": "independent_hpa_evaluation",
        "hpa_labels_used_for_model_or_threshold_selection": False,
        "hpa_path": str(hpa_path),
        "final_output_dir": str(final_output_dir),
        "model_name": model_name,
        "architectures": architectures,
        "fold_models_per_architecture": 5,
        "batch_size_per_device": args.batch_size,
        "mixed_precision": args.mixed_precision,
        "num_hpa_proteins": len(hpa_dataset),
        "evaluated_labels": list(HPA_BENCHMARK_LABELS),
        "hpa_csv_label_support": hpa_label_support,
        "excluded_unannotated_outputs": [
            label for label in DEEPLOC_LABELS if label not in HPA_LABELS
        ],
        "excluded_low_support_hpa_labels": [
            "Lysosome/Vacuole",
            "Peroxisome",
        ],
        "threshold_source": (
            "per_fold_mcc_thresholds_calibrated_on_the_four_swissprot_"
            "training_partitions"
        ),
        "threshold_comparison": "probability > threshold",
        "force_at_least_one_label": False,
        "fold_aggregation_std_ddof": 0,
        "ensemble": (
            "mean fold probability compared with mean fold threshold"
        ),
        "inference_resources_exclude_model_loading": True,
        "runtime": {
            "launcher": (
                "torchrun" if launched_with_torchrun else "python"
            ),
            "world_size": world_size,
            "visible_cuda_devices": torch.cuda.device_count(),
            "used_cuda_device": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "inference_gpu_count": 1,
        },
    }
    (output_root / "hpa_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    all_results = {}
    for architecture in architectures:
        print(f"\nEvaluating HPA: {architecture}", flush=True)
        all_results[architecture] = evaluate_architecture(
            architecture=architecture,
            architecture_dir=final_output_dir / architecture,
            output_dir=output_root / architecture,
            hpa_dataset=hpa_dataset,
            protein_ids=protein_ids,
            labels=labels,
            tokenizer=tokenizer,
            args=args,
        )

    (output_root / "hpa_all_architectures.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "architectures": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nHPA results saved to: {output_root}", flush=True)


if __name__ == "__main__":
    main()
