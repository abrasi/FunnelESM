#!/usr/bin/env python3
"""Run fixed CARE full-train fits and evaluate every model on each test split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ARCHITECTURES = (
    "baseline",
    "mean_layer_6",
    "mean_layers_4_8",
    "pool_query_layer_6",
    "pool_query_layers_4_8",
)
METRICS = (
    "top1_ec1_accuracy",
    "macro_f1",
    "micro_f1",
    "jaccard",
    "accuracy",
    "macro_mcc",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fit each selected CARE architecture on the complete training set "
            "for five seeds, then test each fit on all supplied official splits."
        )
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csvs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pilot_dir")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
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
    if args.pilot_dir:
        if args.learning_rate is None or args.epochs is not None:
            raise ValueError(
                "With --pilot_dir, pass the common --learning_rate; epochs are "
                "selected per architecture from pilot_summary.json."
            )
        pilot_dir = Path(args.pilot_dir).resolve()
        manifest = json.loads(
            (pilot_dir / "pilot_manifest.json").read_text(encoding="utf-8")
        )
        summaries = json.loads(
            (pilot_dir / "pilot_summary.json").read_text(encoding="utf-8")
        )
        rates = sorted(float(value) for value in manifest["learning_rates"])
        at_boundary = len(rates) > 1 and args.learning_rate in (rates[0], rates[-1])
        configs = {}
        problems = []
        for architecture in args.architectures:
            candidates = [
                row for row in summaries
                if row["architecture"] == architecture
                and abs(float(row["learning_rate"]) - args.learning_rate) < 1e-12
            ]
            if not candidates:
                problems.append(f"{architecture}: no run at LR={args.learning_rate:g}")
                continue
            best = min(candidates, key=lambda row: row["best_validation_loss"])
            if not best.get("converged", False):
                problems.append(f"{architecture}: validation loss did not plateau")
            configs[architecture] = {
                "learning_rate": float(args.learning_rate),
                "epochs": int(best["best_epoch"]),
                "pilot_best_validation_loss": float(best["best_validation_loss"]),
            }
        if at_boundary:
            problems.append("the requested common LR is at the pilot search boundary")
        if problems and not args.allow_unready_selection:
            raise ValueError(
                "Pilot configuration is not ready: " + "; ".join(problems)
                + ". Expand the pilot search or explicitly pass --allow_unready_selection."
            )
        return configs, {
            "pilot_dir": str(pilot_dir),
            "learning_rate_source": "common learning rate selected from pilot runs",
            "at_search_boundary": at_boundary,
            "warnings": problems,
        }

    if args.learning_rate is None or args.epochs is None:
        raise ValueError(
            "Pass --pilot_dir plus --learning_rate, or pass both "
            "--learning_rate and --epochs."
        )
    if args.learning_rate <= 0 or args.epochs < 1:
        raise ValueError("learning_rate and epochs must be positive")
    configs = {
        architecture: {
            "learning_rate": float(args.learning_rate),
            "epochs": int(args.epochs),
        }
        for architecture in args.architectures
    }
    return configs, {"learning_rate_source": "manual", "warnings": []}


def mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return {"mean": float(mean), "std": float(std)}


def summarize(records, architectures, test_names):
    result = {}
    for architecture in architectures:
        architecture_runs = [
            run for run in records if run["architecture"] == architecture
        ]
        if not architecture_runs:
            continue
        summary = {
            "n_seeds": len(architecture_runs),
            "seeds": [run["seed"] for run in architecture_runs],
            "learning_rate": architecture_runs[0]["learning_rate"],
            "epochs": architecture_runs[0]["epochs"],
            "train_gpu_energy_kwh": mean_std([
                run["resources"]["train_gpu_energy_kwh"] for run in architecture_runs
            ]),
            "train_time_s": mean_std([
                run["resources"]["train_time_s"] for run in architecture_runs
            ]),
            "peak_gpu_memory_gb": mean_std([
                run["resources"]["peak_gpu_memory_gb"] for run in architecture_runs
            ]),
            "tests": {},
        }
        for test_name in test_names:
            test_summary = {"metrics": {}, "resources": {}}
            metric_names = architecture_runs[0]["test_metrics"][test_name]["metrics"].keys()
            for metric in metric_names:
                test_summary["metrics"][metric] = mean_std([
                    run["test_metrics"][test_name]["metrics"][metric]
                    for run in architecture_runs
                ])
            for resource in ("gpu_energy_kwh", "time_s"):
                test_summary["resources"][resource] = mean_std([
                    run["resources"]["test_by_split"][test_name][resource]
                    for run in architecture_runs
                ])
            summary["tests"][test_name] = test_summary
        result[architecture] = summary
    return result


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.nproc_per_node < 1 or not args.seeds:
        raise ValueError("batch_size, nproc_per_node, and seeds must be positive")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    test_paths = [Path(path).resolve() for path in args.test_csvs]
    test_names = [path.stem for path in test_paths]
    if len(set(test_names)) != len(test_names):
        raise ValueError("Test CSV basenames must be unique")
    hyperparameters, selection_info = resolve_hyperparameters(args)

    root = Path(__file__).resolve().parents[1]
    train_csv = Path(args.train_csv).resolve()
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
                str(root / "scripts" / "train_care_final.py"),
                "--train_csv", str(train_csv),
                "--test_csvs", *[str(path) for path in test_paths],
                "--output_dir", str(run_dir),
                "--architecture", architecture,
                "--epochs", str(config["epochs"]),
                "--batch_size", str(args.batch_size),
                "--learning_rate", str(config["learning_rate"]),
                "--seed", str(seed),
                "--validation_seed", str(args.validation_seed),
                "--validation_size", str(args.validation_size),
                "--max_residues", str(args.max_residues),
                "--focal_gamma", str(args.focal_gamma),
                "--model_name", args.model_name,
                "--mixed_precision", args.mixed_precision,
            ]
            if args.cluster_metadata_csv:
                command.extend([
                    "--cluster_metadata_csv",
                    str(Path(args.cluster_metadata_csv).resolve()),
                ])
            runs.append({
                "architecture": architecture,
                "seed": seed,
                **config,
                "output_dir": str(run_dir),
                "command": command,
            })

    manifest = {
        "purpose": "CARE final: full official train fit, fixed hyperparameters, test all official splits with same model",
        "train_csv": str(train_csv),
        "test_csvs": [str(path) for path in test_paths],
        "test_names": test_names,
        "test_used_for_selection": False,
        "selection": selection_info,
        "hyperparameters": hyperparameters,
        "architectures": list(args.architectures),
        "seeds": list(args.seeds),
        "batch_size_per_device": args.batch_size,
        "effective_train_batch_size": args.batch_size * args.nproc_per_node,
        "cluster_metadata_csv": (
            str(Path(args.cluster_metadata_csv).resolve())
            if args.cluster_metadata_csv else None
        ),
        "focal_gamma": args.focal_gamma,
        "max_residues": args.max_residues,
        "mixed_precision": args.mixed_precision,
        "model_name": args.model_name,
        "nproc_per_node": args.nproc_per_node,
        "final_fit_uses_full_train": True,
        "final_fit_validation": False,
        "runs": runs,
    }
    manifest_path = output / "final_experiment_manifest.json"
    if args.skip_existing and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        compare_keys = (
            "train_csv", "test_csvs", "test_names", "selection", "hyperparameters",
            "architectures", "seeds", "batch_size_per_device", "cluster_metadata_csv",
            "focal_gamma", "max_residues", "mixed_precision", "model_name",
            "nproc_per_node",
        )
        if any(old.get(key) != manifest.get(key) for key in compare_keys):
            raise RuntimeError("Final manifest differs; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Final manifest: {manifest_path}", flush=True)

    records = []
    for run in runs:
        result_path = Path(run["output_dir"]) / "run_result.json"
        print("\n" + " ".join(run["command"]), flush=True)
        if args.dry_run:
            continue
        if args.skip_existing and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            subprocess.run(run["command"], cwd=root, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update({
            "architecture": run["architecture"],
            "seed": run["seed"],
            "learning_rate": run["learning_rate"],
            "epochs": run["epochs"],
        })
        records.append(result)

    if args.dry_run:
        return
    results = {
        "runs": records,
        "summary": summarize(records, args.architectures, test_names),
    }
    results_path = output / "care_final_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Final results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
