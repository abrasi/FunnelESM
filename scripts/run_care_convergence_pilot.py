#!/usr/bin/env python3
"""Run CARE development pilots with torchrun, without loading the test set."""

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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only CARE convergence pilot over the same five "
            "architectures as DeepLoc."
        )
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/care_pilot")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[1e-5, 5e-5, 1e-4])
    parser.add_argument("--architectures", nargs="+", choices=tuple(ARCHITECTURES), default=tuple(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_seed", type=int, default=0)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--max_residues", type=int, default=1022)
    parser.add_argument("--cluster_metadata_csv")
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--model_name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--convergence_patience", type=int, default=3)
    parser.add_argument("--convergence_min_delta", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def lr_name(value):
    return f"{value:.3e}".replace(".", "p").replace("-", "m").replace("+", "")


def main():
    args = build_parser().parse_args()
    if args.epochs < 2 or args.batch_size < 1:
        raise ValueError("epochs must be at least 2 and batch_size positive")
    if args.nproc_per_node < 1:
        raise ValueError("nproc_per_node must be positive")
    if len(args.learning_rates) < 2 or any(lr <= 0 for lr in args.learning_rates):
        raise ValueError("Provide at least two positive learning rates")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        raise ValueError("learning_rates must be unique")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if args.convergence_patience < 1 or args.convergence_patience >= args.epochs:
        raise ValueError("convergence_patience must be in [1, epochs)")
    if args.convergence_min_delta < 0:
        raise ValueError("convergence_min_delta cannot be negative")

    root = Path(__file__).resolve().parents[1]
    train_csv = Path(args.train_csv).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    runs = []
    for architecture in args.architectures:
        for learning_rate in sorted(args.learning_rates):
            name = f"{architecture}_lr_{lr_name(learning_rate)}"
            run_dir = output / name
            command = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                str(root / "scripts" / "train_care.py"),
                "--train_csv", str(train_csv),
                "--output_dir", str(run_dir),
                "--architecture", architecture,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--learning_rate", str(learning_rate),
                "--seed", str(args.seed),
                "--validation_seed", str(args.validation_seed),
                "--validation_size", str(args.validation_size),
                "--max_residues", str(args.max_residues),
                "--focal_gamma", str(args.focal_gamma),
                "--model_name", args.model_name,
                "--mixed_precision", args.mixed_precision,
                "--convergence_pilot",
                *ARCHITECTURES[architecture],
            ]
            if args.cluster_metadata_csv:
                command.extend(["--cluster_metadata_csv", str(Path(args.cluster_metadata_csv).resolve())])
            runs.append({
                "name": name,
                "architecture": architecture,
                "learning_rate": learning_rate,
                "seed": args.seed,
                "output_dir": str(run_dir),
                "command": command,
            })

    manifest = {
        "purpose": "CARE validation-only hyperparameter pilot",
        "train_csv": str(train_csv),
        "test_csv": None,
        "test_loaded_or_evaluated": False,
        "architectures": list(args.architectures),
        "learning_rates": sorted(args.learning_rates),
        "epochs": args.epochs,
        "batch_size_per_device": args.batch_size,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "validation_size": args.validation_size,
        "max_residues": args.max_residues,
        "cluster_metadata_csv": (
            str(Path(args.cluster_metadata_csv).resolve())
            if args.cluster_metadata_csv else None
        ),
        "focal_gamma": args.focal_gamma,
        "mixed_precision": args.mixed_precision,
        "model_name": args.model_name,
        "nproc_per_node": args.nproc_per_node,
        "convergence_patience": args.convergence_patience,
        "convergence_min_delta": args.convergence_min_delta,
        "selection_metric": "minimum eval_loss",
        "runs": runs,
    }
    manifest_path = output / "pilot_manifest.json"
    if args.skip_existing and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = (
            "train_csv", "architectures", "learning_rates", "epochs",
            "batch_size_per_device", "seed", "validation_seed", "validation_size",
            "max_residues", "cluster_metadata_csv", "focal_gamma",
            "mixed_precision", "model_name", "nproc_per_node",
            "convergence_patience", "convergence_min_delta",
        )
        if any(old.get(key) != manifest.get(key) for key in keys):
            raise RuntimeError("Pilot manifest differs from this configuration; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Pilot manifest: {manifest_path}", flush=True)

    for run in runs:
        result_path = Path(run["output_dir"]) / "run_result.json"
        history_path = Path(run["output_dir"]) / "epoch_history.json"
        print("\n" + " ".join(run["command"]), flush=True)
        if args.dry_run:
            continue
        if args.skip_existing and result_path.exists() and history_path.exists():
            print(f"Skipping completed pilot: {run['name']}", flush=True)
            continue
        subprocess.run(run["command"], cwd=root, check=True)

    if not args.dry_run:
        subprocess.run(
            [sys.executable, str(root / "scripts" / "summarize_care_convergence_pilot.py"),
             "--pilot_dir", str(output)],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
