#!/usr/bin/env python3
"""Summarize CARE epoch curves and select LR/epoch budgets per architecture."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "eval_loss", "eval_macro_f1", "eval_micro_f1", "eval_jaccard",
    "eval_accuracy", "eval_num_labels_predicted", "eval_macro_mcc",
    "eval_top1_ec1_accuracy",
)


def convergence_diagnostic(records, patience, min_delta):
    best_loss = float("inf")
    last_improvement = None
    for record in records:
        epoch = int(round(record["epoch"]))
        loss = float(record["eval_loss"])
        if loss < best_loss - min_delta:
            best_loss = loss
            last_improvement = epoch
    last_epoch = int(round(records[-1]["epoch"]))
    return {
        "last_epoch": last_epoch,
        "last_improvement_epoch": last_improvement,
        "epochs_since_improvement": last_epoch - last_improvement,
        "converged": last_epoch - last_improvement >= patience,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", required=True)
    parser.add_argument("--convergence_patience", type=int, default=None)
    parser.add_argument("--convergence_min_delta", type=float, default=None)
    args = parser.parse_args()
    pilot_dir = Path(args.pilot_dir)
    manifest = json.loads((pilot_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    patience = args.convergence_patience or manifest["convergence_patience"]
    min_delta = (
        args.convergence_min_delta
        if args.convergence_min_delta is not None
        else manifest["convergence_min_delta"]
    )
    if patience < 1 or min_delta < 0:
        raise ValueError("Invalid convergence diagnostic options")

    curve_rows = []
    summaries = []
    for run in manifest["runs"]:
        run_dir = Path(run["output_dir"])
        result_path = run_dir / "run_result.json"
        history_path = run_dir / "epoch_history.json"
        if not result_path.exists() or not history_path.exists():
            print(f"Incomplete pilot run, skipped: {run['name']}")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        records = [record for record in history if "eval_loss" in record]
        if not records:
            print(f"No epoch validation records, skipped: {run['name']}")
            continue
        for record in records:
            row = {
                "run": run["name"],
                "architecture": run["architecture"],
                "learning_rate": run["learning_rate"],
                "seed": run["seed"],
                "epoch": record.get("epoch"),
            }
            row.update({key: record.get(key) for key in METRICS})
            curve_rows.append(row)
        best = min(records, key=lambda row: row["eval_loss"])
        summaries.append({
            "run": run["name"],
            "architecture": run["architecture"],
            "learning_rate": float(run["learning_rate"]),
            "seed": run["seed"],
            "best_epoch": int(round(best["epoch"])),
            "best_validation_loss": float(best["eval_loss"]),
            "best_validation_metrics": {
                key: best[key] for key in METRICS if key in best
            },
            **convergence_diagnostic(records, patience, min_delta),
            "resources": result["resources"],
        })

    curves_path = pilot_dir / "learning_curves.csv"
    if curve_rows:
        with curves_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=curve_rows[0].keys())
            writer.writeheader()
            writer.writerows(curve_rows)
    (pilot_dir / "pilot_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    learning_rates = sorted(float(value) for value in manifest["learning_rates"])
    selections = {}
    for architecture in manifest["architectures"]:
        candidates = [row for row in summaries if row["architecture"] == architecture]
        rates_seen = sorted({row["learning_rate"] for row in candidates})
        missing_rates = sorted(set(learning_rates) - set(rates_seen))
        if not candidates:
            selections[architecture] = {
                "ready_for_final": False,
                "warnings": ["No completed runs for this architecture."],
            }
            continue
        best = min(candidates, key=lambda row: row["best_validation_loss"])
        at_boundary = len(learning_rates) > 1 and best["learning_rate"] in (
            learning_rates[0], learning_rates[-1]
        )
        warnings = []
        if not best["converged"]:
            warnings.append("Validation loss has not plateaued; increase the pilot epoch budget.")
        if at_boundary:
            warnings.append("Selected learning rate is at the search boundary; extend the LR grid.")
        if missing_rates:
            warnings.append("Incomplete LR grid: " + ", ".join(map(str, missing_rates)))
        selections[architecture] = {
            "learning_rate": best["learning_rate"],
            "epochs": best["best_epoch"],
            "best_validation_loss": best["best_validation_loss"],
            "development_seed": best["seed"],
            "validation_seed": manifest["validation_seed"],
            "converged": best["converged"],
            "lr_at_search_boundary": at_boundary,
            "evaluated_learning_rates": rates_seen,
            "missing_learning_rates": missing_rates,
            "ready_for_final": not warnings,
            "warnings": warnings,
        }
    selection = {
        "selection_metric": "minimum validation eval_loss",
        "selection_scope": "independently per architecture",
        "convergence_patience": patience,
        "convergence_min_delta": min_delta,
        "architectures": selections,
    }
    selection_path = pilot_dir / "pilot_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    print("\nBest pilot run by architecture and learning rate")
    for row in summaries:
        print(
            f"{row['architecture']:24} lr={row['learning_rate']:.3g} "
            f"epoch={row['best_epoch']:>2} loss={row['best_validation_loss']:.6f} "
            f"converged={row['converged']}"
        )
    print("\nSelected configuration by architecture")
    for architecture, selected in selections.items():
        if "learning_rate" not in selected:
            print(f"{architecture:24} no completed runs")
            continue
        print(
            f"{architecture:24} lr={selected['learning_rate']:.3g} "
            f"epochs={selected['epochs']:>2} ready={selected['ready_for_final']}"
        )
        for warning in selected["warnings"]:
            print(f"  WARNING: {warning}")
    print(f"\nLearning curves: {curves_path}")
    print(f"Pilot summary: {pilot_dir / 'pilot_summary.json'}")
    print(f"Pilot selection: {selection_path}")


if __name__ == "__main__":
    main()
