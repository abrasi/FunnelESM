#!/usr/bin/env python3
"""Launch a CARE convergence pilot or final experiments with manual LR/epochs."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig

from funnel_esm.metrics import summarize_metrics

# Edit these 1-based layer numbers when changing the pretrained model.
# The supplied comparison uses ESM-2 650M (33 layers).
EXPERIMENTS = {
    "baseline": ("train_care_baseline.py", []),
    "mean_layer_16": ("train_funnel.py", [16]),
    "mean_layers_11_22": ("train_funnel.py", [11, 22]),
    "pool_query_layer_16": ("train_funnel_pool_query.py", [16]),
    "pool_query_layers_11_22": ("train_funnel_pool_query.py", [11, 22]),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csvs", nargs="+", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=["baseline", "mean_layer_16"],
    )
    parser.add_argument("--model_name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rates", type=float, nargs="+")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--max_residues", type=int, default=1022)
    parser.add_argument("--cluster_metadata_csv")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def write_final_results(manifest, output):
    """Save individual runs and mean/std directly, as in DeepLoc CV results."""
    records = [
        json.loads((Path(run["output_dir"]) / "run_result.json").read_text())
        for run in manifest["runs"]
    ]
    summary = {}
    for architecture in manifest["config"]["architectures"]:
        runs = [row for row in records if row["architecture"] == architecture]
        training = [
            {key: value for key, value in row["resources"].items() if key != "test_by_split"}
            for row in runs
        ]
        tests = {}
        for split in runs[0]["test_metrics"]:
            tests[split] = {
                "n_examples": runs[0]["test_metrics"][split]["n_examples"],
                "metrics": summarize_metrics(
                    [row["test_metrics"][split]["metrics"] for row in runs]
                ),
                "resources": summarize_metrics(
                    [row["resources"]["test_by_split"][split] for row in runs]
                ),
            }
        summary[architecture] = {"training": summarize_metrics(training), "tests": tests}
        result = {"config": manifest["config"], "runs": runs, "summary": summary[architecture]}
        (output / architecture / "care_results.json").write_text(json.dumps(result, indent=2))
    result = {"config": manifest["config"], "runs": records, "summary": summary}
    (output / "care_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def run_experiments(args):
    pilot = args.phase == "pilot"
    if pilot:
        if args.test_csvs or args.learning_rate is not None:
            raise ValueError("Pilots use --learning_rates and must not receive official tests")
        rates = args.learning_rates or [1e-5, 5e-5, 1e-4]
        seeds = args.seeds or [42]
        epochs = args.epochs if args.epochs is not None else 30
        if len(seeds) != 1:
            raise ValueError("Use one development seed for the pilot")
    else:
        if args.learning_rate is None or args.epochs is None or not args.test_csvs:
            raise ValueError("Final runs require --learning_rate, --epochs and --test_csvs")
        if args.learning_rates is not None or args.cluster_metadata_csv:
            raise ValueError("LR grids and validation metadata belong to the pilot")
        rates, seeds, epochs = [args.learning_rate], args.seeds or [42, 43, 44, 45, 46], args.epochs
    if (
        epochs < 1
        or args.batch_size < 1
        or args.gradient_accumulation_steps < 1
        or args.nproc_per_node < 1
    ):
        raise ValueError("epochs, batch_size, accumulation and GPU processes must be positive")
    if any(rate <= 0 for rate in rates) or len(set(rates)) != len(rates):
        raise ValueError("Learning rates must be positive and unique")
    if len(set(seeds)) != len(seeds) or len(set(args.architectures)) != len(args.architectures):
        raise ValueError("Seeds and architecture names must be unique")
    if not 1 <= args.max_residues <= 1022:
        raise ValueError("max_residues must be between 1 and 1022")
    backbone = AutoConfig.from_pretrained(args.model_name)
    if backbone.model_type != "esm" or backbone.position_embedding_type != "rotary":
        raise ValueError("Choose a pretrained ESM-2 model")
    for architecture in args.architectures:
        _, layers = EXPERIMENTS[architecture]
        if layers != sorted(set(layers)) or any(
            not 1 <= layer < backbone.num_hidden_layers for layer in layers
        ):
            raise ValueError(
                f"{architecture}: layers {layers} are invalid for {backbone.num_hidden_layers} layers. Edit EXPERIMENTS in run_care.py"
            )

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    train = Path(args.train_csv).resolve()
    tests = [Path(path).resolve() for path in args.test_csvs]
    if len({path.stem for path in tests}) != len(tests):
        raise ValueError("Test filenames must be distinct")
    metadata = Path(args.cluster_metadata_csv).resolve() if args.cluster_metadata_csv else None
    config = {
        "phase": args.phase,
        "model_name": args.model_name,
        "n_layers": backbone.num_hidden_layers,
        "hidden_size": backbone.hidden_size,
        "train_csv": str(train),
        "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "test_csvs": [str(path) for path in tests],
        "test_sha256": {path.stem: hashlib.sha256(path.read_bytes()).hexdigest() for path in tests},
        "architectures": args.architectures,
        "pool_after_layers": {name: EXPERIMENTS[name][1] for name in args.architectures},
        "epochs": epochs,
        "learning_rates": sorted(rates),
        "seeds": seeds,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "max_residues": args.max_residues,
        "nproc_per_node": args.nproc_per_node,
        "effective_train_batch_size": args.batch_size
        * args.gradient_accumulation_steps
        * args.nproc_per_node,
        "cluster_metadata_csv": str(metadata) if metadata else None,
        "cluster_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()
        if metadata
        else None,
    }
    runs = []
    for architecture in args.architectures:
        script, layers = EXPERIMENTS[architecture]
        for rate in sorted(rates):
            for seed in seeds:
                name = f"{architecture}_lr_{rate}" if pilot else f"{architecture}/seed_{seed}"
                run_dir = output / name
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={args.nproc_per_node}",
                    str(root / "scripts" / script),
                    "--train_csv",
                    str(train),
                    "--output_dir",
                    str(run_dir),
                    "--architecture",
                    architecture,
                    "--model_name",
                    args.model_name,
                    "--epochs",
                    str(epochs),
                    "--learning_rate",
                    str(rate),
                    "--seed",
                    str(seed),
                    "--batch_size",
                    str(args.batch_size),
                    "--gradient_accumulation_steps",
                    str(args.gradient_accumulation_steps),
                    "--mixed_precision",
                    args.mixed_precision,
                    "--max_residues",
                    str(args.max_residues),
                ]
                if layers:
                    command.extend(["--task", "care", "--pool_after_layers", *map(str, layers)])
                if pilot:
                    command.append("--convergence_pilot")
                if metadata:
                    command.extend(["--cluster_metadata_csv", str(metadata)])
                if tests:
                    command.extend(["--test_csvs", *map(str, tests)])
                runs.append(
                    {
                        "name": name,
                        "architecture": architecture,
                        "learning_rate": rate,
                        "seed": seed,
                        "output_dir": str(run_dir),
                        "command": command,
                    }
                )
    manifest = {"config": config, "runs": runs}
    path = output / ("pilot_manifest.json" if pilot else "final_experiment_manifest.json")
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Campaign configuration/data changed; use a new output_dir")
        if not args.skip_existing and not args.dry_run:
            raise ValueError("Campaign already exists; use --skip_existing or a new output_dir")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output_dir for a new campaign")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    for run in runs:
        print(shlex.join(run["command"]), flush=True)
        if args.dry_run:
            continue
        if args.skip_existing and (Path(run["output_dir"]) / "run_result.json").exists():
            continue
        subprocess.run(run["command"], cwd=root, check=True)
    if not args.dry_run:
        if pilot:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/summarize_care_convergence_pilot.py"),
                    "--pilot_dir",
                    str(output),
                ],
                cwd=root,
                check=True,
            )
        else:
            write_final_results(manifest, output)


if __name__ == "__main__":
    run_experiments(build_parser().parse_args())
