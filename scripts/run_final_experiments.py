#!/usr/bin/env python3
"""Run the fixed five-fold DeepLoc architecture comparison."""

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

PROTOCOL_VERSION = 2


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the five fixed architectures on all five official "
            "DeepLoc 2.0 partitions."
        )
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--output_dir",
        default="outputs/final_deeploc_cv",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip an architecture only when its five-fold results file "
            "already exists."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write the manifest and print commands without executing them.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc_per_node must be positive")

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset_path).resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    runs = []
    for name, script_name, extra_args in EXPERIMENTS:
        run_output_dir = output_root / name
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            str(project_root / "scripts" / script_name),
            "--dataset_path",
            str(dataset_path),
            "--output_dir",
            str(run_output_dir),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--learning_rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
            "--mixed_precision",
            args.mixed_precision,
            "--model_name",
            args.model_name,
            "--folds",
            "0",
            "1",
            "2",
            "3",
            "4",
            *(("--skip_existing_folds",) if args.skip_existing else ()),
            *extra_args,
        ]
        runs.append(
            {
                "name": name,
                "output_dir": str(run_output_dir),
                "command": command,
            }
        )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "final_deeploc_five_fold_architecture_comparison",
        "dataset_path": str(dataset_path),
        "official_outer_evaluation_partitions": [0, 1, 2, 3, 4],
        "outer_training_partitions": (
            "all official partitions except the outer evaluation partition"
        ),
        "checkpoint_selection": {
            "metric": "eval_loss",
            "data": (
                "fixed 2048-example internal validation split from the "
                "four outer training partitions"
            ),
            "internal_validation_seed": 0,
            "maximum_epochs": args.epochs,
            "early_stopping": False,
            "save_total_limit": 1,
        },
        "threshold_calibration": {
            "metric": "per_label_mcc",
            "data": "four official training partitions",
            "outer_evaluation_labels_used": False,
            "force_at_least_one_label": False,
            "comparison": "probability > threshold",
        },
        "fold_aggregation_std_ddof": 0,
        "per_protein_outputs": "outer_fold_predictions.csv",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size_per_device": args.batch_size,
        "effective_train_batch_size": (
            args.batch_size * args.nproc_per_node
        ),
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
        "fold_seed_rule": "seed + validation_partition",
        "model_name": args.model_name,
        "nproc_per_node": args.nproc_per_node,
        "runs": runs,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "final_experiment_manifest.json"
    if args.skip_existing:
        if manifest_path.exists():
            previous_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            comparable_keys = (
                "protocol_version",
                "dataset_path",
                "epochs",
                "learning_rate",
                "batch_size_per_device",
                "mixed_precision",
                "seed",
                "model_name",
                "nproc_per_node",
            )
            if any(
                previous_manifest.get(key) != manifest.get(key)
                for key in comparable_keys
            ):
                raise RuntimeError(
                    "--skip_existing cannot combine results produced with "
                    "a different final protocol. Use another output "
                    "directory."
                )
        elif any(output_root.glob("*/fold_*/fold_result.json")):
            raise RuntimeError(
                "Completed folds exist without a final manifest, so their "
                "configuration cannot be verified. Use a new output "
                "directory."
            )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Final manifest saved to: {manifest_path}", flush=True)

    for run in runs:
        result_path = (
            Path(run["output_dir"])
            / "cross_validation_results.json"
        )
        print("\n" + " ".join(run["command"]), flush=True)
        if args.dry_run:
            continue
        if args.skip_existing and result_path.exists():
            results = json.loads(result_path.read_text(encoding="utf-8"))
            completed_folds = {
                fold_result["fold"]
                for fold_result in results.get("folds", [])
            }
            if completed_folds == set(range(5)):
                print(
                    f"Skipping completed architecture: {run['name']}",
                    flush=True,
                )
                continue

        subprocess.run(
            run["command"],
            cwd=project_root,
            check=True,
        )


if __name__ == "__main__":
    main()
