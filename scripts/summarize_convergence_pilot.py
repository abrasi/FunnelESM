#!/usr/bin/env python3
"""Collect per-epoch metrics produced by the convergence pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EVAL_KEYS = (
    "eval_loss",
    "eval_macro_f1",
    "eval_micro_f1",
    "eval_jaccard",
    "eval_accuracy",
    "eval_num_labels_predicted",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot_dir",
        default="outputs/convergence_pilot",
    )
    return parser


def main():
    args = build_parser().parse_args()
    pilot_dir = Path(args.pilot_dir)
    manifest_path = pilot_dir / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    curve_rows = []
    run_summaries = []

    for run in manifest["runs"]:
        run_dir = Path(run["output_dir"])
        fold = run["fold"]
        history_path = run_dir / f"fold_{fold}" / "epoch_history.json"
        result_path = run_dir / "cross_validation_results.json"

        if not history_path.exists() or not result_path.exists():
            print(f"Incomplete run, skipped: {run['name']}")
            continue

        history = json.loads(history_path.read_text(encoding="utf-8"))
        evaluation_records = [
            record
            for record in history
            if "eval_macro_f1" in record
        ]
        if not evaluation_records:
            print(f"No epoch evaluations found: {run['name']}")
            continue

        for record in evaluation_records:
            row = {
                "run": run["name"],
                "architecture": run["architecture"],
                "learning_rate": run["learning_rate"],
                "fold": fold,
                "epoch": record["epoch"],
            }
            row.update({key: record.get(key) for key in EVAL_KEYS})
            curve_rows.append(row)

        best_record = max(
            evaluation_records,
            key=lambda record: record["eval_macro_f1"],
        )
        final_results = json.loads(
            result_path.read_text(encoding="utf-8")
        )["folds"][0]

        summary = {
            "run": run["name"],
            "architecture": run["architecture"],
            "learning_rate": run["learning_rate"],
            "fold": fold,
            "best_epoch": best_record["epoch"],
        }
        summary.update(
            {key: best_record.get(key) for key in EVAL_KEYS}
        )
        summary.update(
            {
                "train_gpu_energy_j": final_results[
                    "train_gpu_energy_j"
                ],
                "train_time_s": final_results["train_time_s"],
                "peak_gpu_memory_gb": final_results[
                    "peak_gpu_memory_gb"
                ],
            }
        )
        run_summaries.append(summary)

    curves_path = pilot_dir / "learning_curves.csv"
    if curve_rows:
        with curves_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=curve_rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(curve_rows)

    summary_path = pilot_dir / "pilot_summary.json"
    summary_path.write_text(
        json.dumps(run_summaries, indent=2),
        encoding="utf-8",
    )

    print("\nBest macro-F1 by run")
    for result in run_summaries:
        print(
            f"{result['architecture']:24} "
            f"lr={result['learning_rate']:.3g} "
            f"epoch={result['best_epoch']:>4} "
            f"macro_f1={result['eval_macro_f1']:.6f}"
        )

    print(f"\nLearning curves saved to: {curves_path}")
    print(f"Pilot summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
