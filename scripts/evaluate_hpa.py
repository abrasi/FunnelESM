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
from funnel_esm.deeploc import DEEPLOC_LABELS, HPA_LABELS
from funnel_esm.metrics import apply_thresholds, label_metric_name, multilabel_metrics


# DeepLoc 2.0 Table 2 evaluates six HPA locations. The downloaded HPA file
# also contains Lysosome/Vacuole and Peroxisome, but with only 2 and 7
# positive examples, respectively; they are excluded from the benchmark.
HPA_BENCHMARK_LABELS = tuple(
    label for label in HPA_LABELS if label not in {"Lysosome/Vacuole", "Peroxisome"}
)
HPA_OUTPUT_INDICES = tuple(
    DEEPLOC_LABELS.index(label) for label in HPA_BENCHMARK_LABELS
)
HPA_DATASET_INDICES = tuple(HPA_LABELS.index(label) for label in HPA_BENCHMARK_LABELS)

METRIC_NAMES = (
    "macro_f1",
    "micro_f1",
    "jaccard",
    "accuracy",
    "num_labels_predicted",
    "macro_mcc",
    *tuple(f"mcc_{label_metric_name(label)}" for label in HPA_BENCHMARK_LABELS),
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final_output_dir", required=True)
    parser.add_argument("--hpa_path", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--precision", choices=("float32", "bf16", "fp16"), default="float32"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max_residues", type=int, default=1022)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no_energy", action="store_true", help="CPU smoke checks: energy is null"
    )
    return parser


def load_thresholds(fold_result_path):
    fold_result = json.loads(fold_result_path.read_text(encoding="utf-8"))
    threshold_dict = fold_result.get("thresholds")
    if threshold_dict is None:
        raise RuntimeError(f"No calibrated thresholds in {fold_result_path}")
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
    label_names = [label_metric_name(label) for label in HPA_BENCHMARK_LABELS]
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
                row[f"prob_{name}"] = float(probabilities[row_index, label_index])
                row[f"true_{name}"] = int(labels[row_index, label_index])
                row[f"pred_{name}"] = int(predictions[row_index, label_index])
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


def evaluate_architecture(
    architecture, architecture_dir, output_dir, hpa_dataset, protein_ids, labels, args
):
    """Sequential five-model ensemble. Weight loading is outside measured windows."""
    from funnel_esm.inference import Meter, run_passes, average_resources
    from funnel_esm.modeling_classification import (
        load_saved_classifier,
        ProteinPredictor,
    )

    meter = Meter(args.device, args.no_energy)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, probabilities, thresholds_all = [], [], []
    signature = None
    for fold in range(5):
        model_dir = architecture_dir / f"fold_{fold}"
        thresholds_10 = load_thresholds(model_dir / "fold_result.json")
        thresholds = thresholds_10[list(HPA_OUTPUT_INDICES)]
        model, fold_tokenizer, _ = load_saved_classifier(
            model_dir, DEEPLOC_LABELS, meter.device, "eager"
        )
        current = (
            model.config.num_hidden_layers,
            model.config.hidden_size,
            tuple(getattr(model.config, "funnel_pool_after_layers", [])),
            getattr(model.config, "funnel_pooling_type", None),
        )
        if signature is not None and current != signature:
            raise ValueError("Fold models have different architectures")
        signature = current
        saved = getattr(model.config, "deeploc_thresholds", None)
        if saved is not None and not np.allclose(
            saved, thresholds_10, atol=1e-12, rtol=0
        ):
            raise ValueError("Checkpoint thresholds differ from fold_result.json")
        predictor = ProteinPredictor(model, fold_tokenizer, args, meter.device)
        metrics, all_probabilities, resource = run_passes(
            predictor,
            hpa_dataset,
            lambda values: compute_hpa_metrics(
                labels, values[:, list(HPA_OUTPUT_INDICES)], thresholds
            ),
            args,
            meter,
            f"hpa_{architecture}_fold_{fold}",
        )
        fold_probs = all_probabilities[:, list(HPA_OUTPUT_INDICES)]
        predictions = apply_thresholds(
            fold_probs, thresholds, force_at_least_one=False, inclusive=False
        )
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        metrics.update(
            fold=fold,
            resources=resource,
            inference_gpu_energy_j=resource["gpu_energy_j"],
            inference_gpu_energy_kwh=resource["gpu_energy_kwh"],
            inference_time_s=resource["time_s"],
            peak_gpu_memory_gb=resource["peak_gpu_memory_gb"],
            joules_per_protein=resource["joules_per_protein"],
            proteins_per_second=resource["proteins_per_second"],
            thresholds=dict(zip(HPA_BENCHMARK_LABELS, thresholds.tolist())),
        )
        save_predictions(
            fold_dir / "hpa_predictions.csv",
            protein_ids,
            fold_probs,
            labels,
            predictions,
        )
        (fold_dir / "hpa_result.json").write_text(json.dumps(metrics, indent=2))
        folds.append(metrics)
        probabilities.append(fold_probs)
        thresholds_all.append(thresholds)
        del model, predictor
        gc.collect()
        torch.cuda.empty_cache()
    ensemble_rows = []
    for repeat in range(args.repeats):
        meter.begin(f"hpa_{architecture}_aggregate_{repeat}")
        ensemble_probs = np.mean(probabilities, axis=0)
        ensemble_thresholds = np.mean(thresholds_all, axis=0)
        ensemble_predictions = apply_thresholds(
            ensemble_probs,
            ensemble_thresholds,
            force_at_least_one=False,
            inclusive=False,
        )
        ensemble = compute_hpa_metrics(labels, ensemble_probs, ensemble_thresholds)
        aggregation = meter.end(allow_zero=True)
        windows = [fold["resources"]["raw_measurements"][repeat] for fold in folds] + [
            aggregation
        ]
        energy = [window["gpu_energy_j"] for window in windows]
        memory = [window["peak_gpu_memory_gb"] for window in windows]
        joules = None if any(value is None for value in energy) else sum(energy)
        ensemble_rows.append(
            dict(
                time_s=sum(window["time_s"] for window in windows),
                gpu_energy_j=joules,
                gpu_energy_kwh=None if joules is None else joules / 3600000,
                peak_gpu_memory_gb=None
                if any(value is None for value in memory)
                else max(memory),
                n_predictions=len(hpa_dataset),
            )
        )
    resource = average_resources(ensemble_rows)
    resource.update(
        aggregation_energy_note="short CPU aggregation may be below GPU energy counter resolution",
        n_models=5,
        n_model_forwards=len(hpa_dataset) * args.repeats * 5,
        included=[
            "five sequence-to-probability forwards",
            "per-fold scoring",
            "ensemble aggregation and scoring",
        ],
        excluded=["weight loading", "CSV reading", "warmup", "serialization"],
    )
    ensemble.update(
        resources=resource,
        inference_gpu_energy_j=resource["gpu_energy_j"],
        inference_gpu_energy_kwh=resource["gpu_energy_kwh"],
        inference_time_s=resource["time_s"],
        peak_gpu_memory_gb=resource["peak_gpu_memory_gb"],
        joules_per_protein=resource["joules_per_protein"],
        proteins_per_second=resource["proteins_per_second"],
        thresholds=dict(zip(HPA_BENCHMARK_LABELS, ensemble_thresholds.tolist())),
    )
    save_predictions(
        output_dir / "hpa_ensemble_predictions.csv",
        protein_ids,
        ensemble_probs,
        labels,
        ensemble_predictions,
    )
    from funnel_esm.metrics import summarize_metrics

    result = dict(
        architecture=architecture,
        evaluated_labels=list(HPA_BENCHMARK_LABELS),
        folds=folds,
        fold_summary=summarize_metrics(
            folds,
            (
                *METRIC_NAMES,
                "inference_gpu_energy_j",
                "inference_time_s",
                "peak_gpu_memory_gb",
                "joules_per_protein",
                "proteins_per_second",
            ),
        ),
        ensemble=ensemble,
        ensemble_definition=dict(
            probabilities="mean across five folds",
            thresholds="mean stored fold thresholds",
            comparison=">",
            resource_accounting="sum five forward windows plus measured ensemble aggregation; N inputs per pass",
        ),
    )
    (output_dir / "hpa_results.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    args = build_parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_with_torchrun = "LOCAL_RANK" in os.environ
    if world_size != 1:
        raise RuntimeError("Inference uses one process/GPU")
    if not args.no_energy and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Zeus; CPU smoke checks use --no_energy"
        )
    if (
        args.batch_size < 1
        or args.repeats < 1
        or args.warmup < 0
        or not 1 <= args.max_residues <= 1022
    ):
        raise ValueError("Invalid benchmark configuration")
    if torch.device(args.device).type == "cpu" and args.precision == "fp16":
        raise ValueError("CPU fp16 is unsupported")
    args.progress = False
    final_output_dir = Path(args.final_output_dir).resolve()
    final_manifest_path = final_output_dir / "final_experiment_manifest.json"
    if not final_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing final experiment manifest: {final_manifest_path}"
        )
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if final_manifest.get("protocol_version") != 2:
        raise RuntimeError("The final results were not produced by protocol version 2.")
    model_name = final_manifest["model_name"]
    manifest_architectures = [run["name"] for run in final_manifest.get("runs", [])]
    if not manifest_architectures or len(set(manifest_architectures)) != len(
        manifest_architectures
    ):
        raise RuntimeError("The final manifest must contain unique experiment names")
    architectures = manifest_architectures

    hpa_path = Path(args.hpa_path).resolve()
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else final_output_dir / "hpa_evaluation"
    )
    from funnel_esm.inference import new_output

    output_root = new_output(output_root)

    import pandas as pd
    from funnel_esm.care import normalize_sequence

    frame = pd.read_csv(hpa_path)
    required = {"sid", "fasta", *HPA_LABELS}
    if not required.issubset(frame.columns):
        raise ValueError("HPA columns missing")
    hpa_dataset = frame.fasta.map(normalize_sequence).tolist()
    protein_ids = frame.sid.astype(str).tolist()
    all_hpa_labels = frame[list(HPA_LABELS)].to_numpy(dtype=np.float64)
    if not np.isin(all_hpa_labels, [0, 1]).all():
        raise ValueError("HPA labels must be binary")
    all_hpa_labels = all_hpa_labels.astype(np.int64)
    labels = all_hpa_labels[:, list(HPA_DATASET_INDICES)]
    hpa_label_support = dict(
        zip(HPA_LABELS, all_hpa_labels.sum(axis=0).astype(int).tolist())
    )
    manifest = {
        "purpose": "independent_hpa_evaluation",
        "hpa_labels_used_for_model_or_threshold_selection": False,
        "hpa_path": str(hpa_path),
        "final_output_dir": str(final_output_dir),
        "model_name": model_name,
        "architectures": architectures,
        "fold_models_per_architecture": 5,
        "batch_size_per_device": args.batch_size,
        "precision": args.precision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_residues": args.max_residues,
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
            "per_fold_mcc_thresholds_calibrated_on_the_four_swissprot_training_partitions"
        ),
        "threshold_comparison": "probability > threshold",
        "force_at_least_one_label": False,
        "fold_aggregation_std_ddof": 0,
        "ensemble": ("mean fold probability compared with mean fold threshold"),
        "inference_resources_exclude_model_loading": True,
        "runtime": {
            "launcher": ("torchrun" if launched_with_torchrun else "python"),
            "world_size": world_size,
            "visible_cuda_devices": torch.cuda.device_count(),
            "used_cuda_device": str(args.device),
            "gpu_name": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else None,
            "inference_gpu_count": int(torch.device(args.device).type == "cuda"),
        },
    }
    from funnel_esm.inference import Meter, runtime

    meter = Meter(args.device, args.no_energy)
    manifest["measurement_runtime"] = runtime(meter.device, args.precision)
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
            args=args,
        )

    from funnel_esm.inference import PROTOCOL, combine_resources

    benchmark_resources = combine_resources(
        [result["ensemble"]["resources"] for result in all_results.values()]
    )
    (output_root / "hpa_all_architectures.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "architectures": all_results,
                "benchmark_resources": benchmark_resources,
                "measurement_protocol": PROTOCOL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nHPA results saved to: {output_root}", flush=True)


if __name__ == "__main__":
    main()
