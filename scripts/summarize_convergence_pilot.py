#!/usr/bin/env python3
"""Summarize curves and select LR/fixed epochs on the development fold."""

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
    parser.add_argument(
        "--convergence_patience",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--convergence_min_delta",
        type=float,
        default=None,
    )
    return parser


def convergence_diagnostic(
    evaluation_records,
    patience,
    min_delta,
):
    """Diagnose convergence after a complete, non-stopped training run."""
    best_so_far = float("inf")
    last_improvement_epoch = None

    for record in evaluation_records:
        loss = float(record["eval_loss"])
        epoch = int(round(record["epoch"]))
        if loss < best_so_far - min_delta:
            best_so_far = loss
            last_improvement_epoch = epoch

    last_epoch = int(round(evaluation_records[-1]["epoch"]))
    epochs_since_improvement = last_epoch - last_improvement_epoch
    return {
        "last_epoch": last_epoch,
        "last_improvement_epoch": last_improvement_epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "converged": epochs_since_improvement >= patience,
    }


def main():
    args = build_parser().parse_args()
    pilot_dir = Path(args.pilot_dir)
    manifest_path = pilot_dir / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patience = (
        args.convergence_patience
        if args.convergence_patience is not None
        else manifest.get("convergence_patience", 3)
    )
    min_delta = (
        args.convergence_min_delta
        if args.convergence_min_delta is not None
        else manifest.get("convergence_min_delta", 0.0)
    )
    if patience < 1:
        raise ValueError("--convergence_patience must be positive")
    if min_delta < 0:
        raise ValueError("--convergence_min_delta cannot be negative")

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

        best_record = min(
            evaluation_records,
            key=lambda record: record["eval_loss"],
        )
        diagnostic = convergence_diagnostic(
            evaluation_records,
            patience,
            min_delta,
        )
        final_results = json.loads(
            result_path.read_text(encoding="utf-8")
        )["folds"][0]

        summary = {
            "run": run["name"],
            "architecture": run["architecture"],
            "learning_rate": run["learning_rate"],
            "fold": fold,
            "best_epoch": int(round(best_record["epoch"])),
            **diagnostic,
        }
        summary.update(
            {key: best_record.get(key) for key in EVAL_KEYS}
        )
        summary.update(
            {
                "pilot_train_gpu_energy_j": final_results[
                    "train_gpu_energy_j"
                ],
                "pilot_train_time_s": final_results["train_time_s"],
                "pilot_peak_gpu_memory_gb": final_results[
                    "peak_gpu_memory_gb"
                ],
            }
        )
        if "threshold_calibration_gpu_energy_j" in final_results:
            summary.update(
                {
                    "pilot_threshold_calibration_gpu_energy_j":
                        final_results[
                            "threshold_calibration_gpu_energy_j"
                        ],
                    "pilot_threshold_calibration_time_s":
                        final_results[
                            "threshold_calibration_time_s"
                        ],
                    "pilot_pipeline_gpu_energy_j": final_results[
                        "pipeline_gpu_energy_j"
                    ],
                    "pilot_pipeline_time_s": final_results[
                        "pipeline_time_s"
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

    learning_rates = sorted(
        float(value) for value in manifest["learning_rates"]
    )
    architectures = []
    for run in manifest["runs"]:
        if run["architecture"] not in architectures:
            architectures.append(run["architecture"])

    selections = {}
    for architecture in architectures:
        candidates = [
            run
            for run in run_summaries
            if run["architecture"] == architecture
        ]
        evaluated_learning_rates = sorted(
            float(run["learning_rate"]) for run in candidates
        )
        missing_learning_rates = sorted(
            set(learning_rates) - set(evaluated_learning_rates)
        )
        if not candidates:
            selections[architecture] = {
                "ready_for_final_cv": False,
                "warnings": [
                    "No completed runs are available for this architecture."
                ],
            }
            continue

        selected = min(
            candidates,
            key=lambda run: run["eval_loss"],
        )
        learning_rate = float(selected["learning_rate"])
        lr_at_boundary = (
            len(learning_rates) > 1
            and learning_rate in (
                learning_rates[0],
                learning_rates[-1],
            )
        )
        ready = (
            selected["converged"]
            and not lr_at_boundary
            and not missing_learning_rates
        )

        warnings = []
        if not selected["converged"]:
            warnings.append(
                "Validation loss has not plateaued for the requested "
                "number of epochs; extend the pilot before final CV."
            )
        if lr_at_boundary:
            warnings.append(
                "The selected LR is on the search-grid boundary; expand "
                "the grid in that direction before final CV."
            )
        if missing_learning_rates:
            warnings.append(
                "The LR grid is incomplete. Missing runs: "
                + ", ".join(
                    f"{learning_rate:.3g}"
                    for learning_rate in missing_learning_rates
                )
            )

        selections[architecture] = {
            "learning_rate": learning_rate,
            "epochs": selected["best_epoch"],
            "best_validation_loss": selected["eval_loss"],
            "development_fold": selected["fold"],
            "seed": manifest["seed"],
            "converged": selected["converged"],
            "lr_at_search_boundary": lr_at_boundary,
            "evaluated_learning_rates": evaluated_learning_rates,
            "missing_learning_rates": missing_learning_rates,
            "ready_for_final_cv": ready,
            "warnings": warnings,
        }

    selection_path = pilot_dir / "pilot_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_metric": "eval_loss",
                "convergence_patience": patience,
                "convergence_min_delta": min_delta,
                "architectures": selections,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nBest validation loss by run")
    for result in run_summaries:
        print(
            f"{result['architecture']:24} "
            f"lr={result['learning_rate']:.3g} "
            f"epoch={result['best_epoch']:>4} "
            f"loss={result['eval_loss']:.6f} "
            f"macro_f1={result['eval_macro_f1']:.6f} "
            f"converged={result['converged']}"
        )

    print("\nSelected fixed configuration by architecture")
    for architecture, selection in selections.items():
        if "learning_rate" not in selection:
            print(f"{architecture:24} no completed runs")
            for warning in selection["warnings"]:
                print(f"  WARNING: {warning}")
            continue
        print(
            f"{architecture:24} "
            f"lr={selection['learning_rate']:.3g} "
            f"epochs={selection['epochs']:>2} "
            f"ready={selection['ready_for_final_cv']}"
        )
        for warning in selection["warnings"]:
            print(f"  WARNING: {warning}")

    print(f"\nLearning curves saved to: {curves_path}")
    print(f"Pilot summary saved to: {summary_path}")
    print(f"Pilot selection saved to: {selection_path}")


if __name__ == "__main__":
    main()
