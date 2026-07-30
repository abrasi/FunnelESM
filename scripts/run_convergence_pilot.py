#!/usr/bin/env python3
"""Run the fixed-budget DeepLoc convergence pilot with torchrun."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = (
    ("baseline", "train_baseline.py", ()),
    ("mean_layer_6", "train_funnel.py", ("--pool_after_layers", "6")),
    (
        "mean_layers_4_8",
        "train_funnel.py",
        ("--pool_after_layers", "4", "8"),
    ),
    (
        "pool_query_layer_6",
        "train_funnel_pool_query.py",
        ("--pool_after_layers", "6"),
    ),
    (
        "pool_query_layers_4_8",
        "train_funnel_pool_query.py",
        ("--pool_after_layers", "4", "8"),
    ),
)

PROTOCOL_VERSION = 3


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline, mean-pooling and pool-query-only convergence "
            "pilots on one DeepLoc fold."
        )
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/convergence_pilot",
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--base_learning_rate",
        type=float,
        default=5e-5,
        help=(
            "Central learning rate. The pilot also uses LR/3 and 3*LR."
        ),
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit LR grid. When provided, it replaces LR/3, LR and "
            "3*LR."
        ),
    )
    parser.add_argument(
        "--convergence_patience",
        type=int,
        default=3,
        help=(
            "Post-hoc diagnostic: require this many final epochs without "
            "a validation-loss improvement. It does not stop training."
        ),
    )
    parser.add_argument(
        "--convergence_min_delta",
        type=float,
        default=0.0,
        help=(
            "Minimum validation-loss decrease counted as an improvement "
            "by the post-hoc convergence diagnostic."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--model_name",
        default="facebook/esm2_t12_35M_UR50D",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of GPU processes passed to torchrun.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs that already contain cross_validation_results.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser


def learning_rate_name(learning_rate):
    return (
        f"{learning_rate:.3e}"
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "")
    )


def main():
    args = build_parser().parse_args()
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2")
    if args.convergence_patience < 1:
        raise ValueError("--convergence_patience must be positive")
    if args.convergence_patience >= args.epochs:
        raise ValueError(
            "--convergence_patience must be smaller than --epochs"
        )
    if args.convergence_min_delta < 0:
        raise ValueError("--convergence_min_delta cannot be negative")

    project_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    if args.learning_rates is None:
        learning_rates = [
            args.base_learning_rate / 3,
            args.base_learning_rate,
            args.base_learning_rate * 3,
        ]
    else:
        learning_rates = args.learning_rates

    if any(learning_rate <= 0 for learning_rate in learning_rates):
        raise ValueError("All learning rates must be positive")
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("Learning rates cannot contain duplicates")
    learning_rates = sorted(learning_rates)

    runs = []

    for experiment_name, script_name, extra_args in EXPERIMENTS:
        for learning_rate in learning_rates:
            run_name = (
                f"{experiment_name}_lr_"
                f"{learning_rate_name(learning_rate)}"
            )
            run_output_dir = output_root / run_name

            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                str(project_root / "scripts" / script_name),
                "--dataset_path",
                str(Path(args.dataset_path).resolve()),
                "--output_dir",
                str(run_output_dir),
                "--batch_size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--learning_rate",
                str(learning_rate),
                "--seed",
                str(args.seed),
                "--mixed_precision",
                args.mixed_precision,
                "--model_name",
                args.model_name,
                "--folds",
                str(args.fold),
                "--eval_each_epoch",
                "--convergence_pilot",
                *extra_args,
            ]

            run_record = {
                "name": run_name,
                "architecture": experiment_name,
                "learning_rate": learning_rate,
                "fold": args.fold,
                "output_dir": str(run_output_dir),
                "command": command,
            }
            runs.append(run_record)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "select_one_common_learning_rate_and_verify_the_shared_"
            "epoch_budget"
        ),
        "selection_metric": "eval_loss",
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "base_learning_rate": args.base_learning_rate,
        "learning_rates": learning_rates,
        "fold": args.fold,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "convergence_patience": args.convergence_patience,
        "convergence_min_delta": args.convergence_min_delta,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
        "model_name": args.model_name,
        "runs": runs,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "pilot_manifest.json"
    if args.skip_existing and manifest_path.exists():
        previous_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        comparable_keys = (
            "protocol_version",
            "dataset_path",
            "learning_rates",
            "fold",
            "batch_size",
            "epochs",
            "mixed_precision",
            "seed",
            "model_name",
        )
        if any(
            previous_manifest.get(key) != manifest.get(key)
            for key in comparable_keys
        ):
            raise RuntimeError(
                "--skip_existing cannot reuse results produced by a "
                "different pilot configuration. Use a new output directory."
            )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Pilot manifest saved to: {manifest_path}", flush=True)

    for run_record in runs:
        command = run_record["command"]
        result_path = (
            Path(run_record["output_dir"])
            / "cross_validation_results.json"
        )
        history_path = (
            Path(run_record["output_dir"])
            / f"fold_{args.fold}"
            / "epoch_history.json"
        )

        print("\n" + " ".join(command), flush=True)
        if args.dry_run:
            continue
        if (
            args.skip_existing
            and result_path.exists()
            and history_path.exists()
        ):
            print(
                f"Skipping completed run: {run_record['name']}",
                flush=True,
            )
            continue

        subprocess.run(
            command,
            cwd=project_root,
            check=True,
        )

    if not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(
                    project_root
                    / "scripts"
                    / "summarize_convergence_pilot.py"
                ),
                "--pilot_dir",
                str(output_root),
                "--convergence_patience",
                str(args.convergence_patience),
                "--convergence_min_delta",
                str(args.convergence_min_delta),
            ],
            cwd=project_root,
            check=True,
        )


if __name__ == "__main__":
    main()
