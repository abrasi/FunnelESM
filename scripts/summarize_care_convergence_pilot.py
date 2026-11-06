#!/usr/bin/env python3
"""Collect CARE pilot learning curves; final hyperparameters remain manual."""

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(pilot_dir):
    manifest = json.loads((pilot_dir / "pilot_manifest.json").read_text())
    curves, summaries = [], []
    for run in manifest["runs"]:
        directory = Path(run["output_dir"])
        if not (directory / "run_result.json").exists():
            print(f"Incomplete pilot, skipped: {run['name']}")
            continue
        result = json.loads((directory / "run_result.json").read_text())
        history = json.loads((directory / "epoch_history.json").read_text())
        evaluations = [row for row in history if "eval_loss" in row]
        info = {
            "run": run["name"],
            "architecture": run["architecture"],
            "learning_rate": run["learning_rate"],
            "seed": run["seed"],
        }
        for row in evaluations:
            curves.append(
                {
                    **info,
                    "epoch": row["epoch"],
                    **{key: value for key, value in row.items() if key.startswith("eval_")},
                }
            )
        best = min(evaluations, key=lambda row: row["eval_loss"])
        summaries.append(
            {
                **info,
                "best_epoch": best["epoch"],
                "last_epoch": evaluations[-1]["epoch"],
                **{key: value for key, value in best.items() if key.startswith("eval_")},
                **result["resources"],
            }
        )
    if not summaries:
        raise ValueError("No completed pilot runs")
    pd.DataFrame(curves).to_csv(pilot_dir / "learning_curves.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(pilot_dir / "pilot_summary.csv", index=False)
    summary.to_json(pilot_dir / "pilot_summary.json", orient="records", indent=2)
    print(
        summary[
            ["architecture", "learning_rate", "best_epoch", "eval_loss", "eval_top1_ec1_accuracy"]
        ].to_string(index=False)
    )
    print("Choose final LR and epochs yourself using these validation curves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_dir", required=True)
    summarize(Path(parser.parse_args().pilot_dir))
