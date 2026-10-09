#!/usr/bin/env python3
"""Run the fixed CARE train/test comparison with torchrun and Zeus metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ARCHITECTURES = {
    "baseline": (),
    "mean_layer_6": (),
    "mean_layers_4_8": (),
    "pool_query_layer_6": (),
    "pool_query_layers_4_8": (),
}
METRICS = (
    "macro_f1", "micro_f1", "jaccard", "accuracy", "num_labels_predicted",
    "macro_mcc", "top1_ec1_accuracy",
    *(f"mcc_ec{i}" for i in range(1, 8)),
)
RESOURCES = (
    "train_gpu_energy_j", "train_gpu_energy_kwh", "train_time_s",
    "threshold_calibration_gpu_energy_j", "threshold_calibration_gpu_energy_kwh",
    "threshold_calibration_time_s", "pipeline_gpu_energy_j",
    "pipeline_gpu_energy_kwh", "pipeline_time_s", "test_gpu_energy_j",
    "test_gpu_energy_kwh", "test_time_s", "peak_gpu_memory_gb",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen ESM-2 baseline and FunnelESM variants on "
            "the official fixed CARE test split."
        )
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pilot_selection_json")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--architectures", nargs="+", choices=tuple(ARCHITECTURES), default=tuple(ARCHITECTURES))
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validation_seed", type=int, default=0)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--max_residues", type=int, default=1022)
    parser.add_argument("--cluster_metadata_csv")
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--model_name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--allow_unready_selection", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def resolve_hyperparameters(args):
    if args.pilot_selection_json:
        if args.learning_rate is not None or args.epochs is not None:
            raise ValueError("Use pilot_selection_json or manual LR/epochs, not both")
        selection_path = Path(args.pilot_selection_json).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        chosen = selection.get("architectures", {})
        result = {}
        for architecture in args.architectures:
            item = chosen.get(architecture)
            if item is None or "learning_rate" not in item or "epochs" not in item:
                raise ValueError(f"No pilot configuration for {architecture}")
            if not item.get("ready_for_final", False) and not args.allow_unready_selection:
                raise ValueError(
                    f"Pilot selection for {architecture} is not marked ready; "
                    "expand the pilot or pass --allow_unready_selection."
                )
            result[architecture] = {
                "learning_rate": float(item["learning_rate"]),
                "epochs": int(item["epochs"]),
            }
        return result
    if args.learning_rate is None or args.epochs is None:
        raise ValueError(
            "Pass --pilot_selection_json or both --learning_rate and --epochs"
        )
    if args.learning_rate <= 0 or args.epochs < 1:
        raise ValueError("learning_rate and epochs must be positive")
    return {
        architecture: {"learning_rate": args.learning_rate, "epochs": args.epochs}
        for architecture in args.architectures
    }


def mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return {"mean": float(mean), "std": float(std)}


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.nproc_per_node < 1 or not args.seeds:
        raise ValueError("batch_size, nproc_per_node, and seeds must be positive")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    hyperparameters = resolve_hyperparameters(args)

    root = Path(__file__).resolve().parents[1]
    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    runs = []
    for architecture in args.architectures:
        config = hyperparameters[architecture]
        for seed in args.seeds:
            run_dir = output / architecture / f"seed_{seed}"
            command = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                str(root / "scripts" / "train_care.py"),
                "--train_csv", str(train_csv), "--test_csv", str(test_csv),
                "--output_dir", str(run_dir), "--architecture", architecture,
                "--epochs", str(config["epochs"]),
                "--batch_size", str(args.batch_size),
                "--learning_rate", str(config["learning_rate"]),
                "--seed", str(seed), "--validation_seed", str(args.validation_seed),
                "--validation_size", str(args.validation_size),
                "--max_residues", str(args.max_residues),
                "--focal_gamma", str(args.focal_gamma),
                "--model_name", args.model_name,
                "--mixed_precision", args.mixed_precision,
            ]
            if args.cluster_metadata_csv:
                command.extend(["--cluster_metadata_csv", str(Path(args.cluster_metadata_csv).resolve())])
            runs.append({
                "architecture": architecture,
                "seed": seed,
                **config,
                "output_dir": str(run_dir),
                "command": command,
            })

    manifest = {
        "purpose": "CARE fixed official train/test evaluation; no cross-validation",
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "pilot_selection_json": (
            str(Path(args.pilot_selection_json).resolve())
            if args.pilot_selection_json else None
        ),
        "hyperparameters": hyperparameters,
        "architectures": list(args.architectures),
        "seeds": list(args.seeds),
        "batch_size_per_device": args.batch_size,
        "effective_train_batch_size": args.batch_size * args.nproc_per_node,
        "validation_seed": args.validation_seed,
        "validation_size": args.validation_size,
        "cluster_metadata_csv": (
            str(Path(args.cluster_metadata_csv).resolve())
            if args.cluster_metadata_csv else None
        ),
        "focal_gamma": args.focal_gamma,
        "max_residues": args.max_residues,
        "mixed_precision": args.mixed_precision,
        "model_name": args.model_name,
        "nproc_per_node": args.nproc_per_node,
        "test_used_for_selection": False,
        "runs": runs,
    }
    manifest_path = output / "final_experiment_manifest.json"
    if args.skip_existing and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = (
            "train_csv", "test_csv", "pilot_selection_json", "hyperparameters",
            "architectures", "seeds", "batch_size_per_device", "validation_seed",
            "validation_size", "cluster_metadata_csv", "focal_gamma",
            "max_residues", "mixed_precision", "model_name", "nproc_per_node",
        )
        if any(old.get(key) != manifest.get(key) for key in keys):
            raise RuntimeError("Final manifest differs from this configuration; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Final manifest: {manifest_path}", flush=True)

    grouped = {architecture: [] for architecture in args.architectures}
    for run in runs:
        result_path = Path(run["output_dir"]) / "run_result.json"
        print("\n" + " ".join(run["command"]), flush=True)
        if args.dry_run:
            continue
        if args.skip_existing and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"Reusing completed run: {run['output_dir']}", flush=True)
        else:
            subprocess.run(run["command"], cwd=root, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        grouped[run["architecture"]].append(result)

    if args.dry_run:
        return
    summary = {}
    for architecture, records in grouped.items():
        if not records:
            continue
        architecture_summary = {
            "n_seeds": len(records),
            "seeds": [record["seed"] for record in records],
        }
        for key in METRICS:
            architecture_summary[key] = mean_std(
                [float(record["test_metrics"][key]) for record in records]
            )
        for key in RESOURCES:
            architecture_summary[key] = mean_std(
                [float(record["resources"][key]) for record in records]
            )
        summary[architecture] = architecture_summary
    results = {"runs": grouped, "summary": summary}
    results_path = output / "care_final_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Final results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
