#!/usr/bin/env python3
"""Standalone CARE inference: no dependency on the CLEAN project."""

import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from funnel_esm.care import CARE_LABELS, read_care, check_disjoint
from funnel_esm.inference import (
    Meter,
    PROTOCOL,
    runtime,
    sha256,
    digest,
    new_output,
    save_json,
    run_passes,
    combine_resources,
)
from funnel_esm.modeling_classification import load_saved_classifier, ProteinPredictor
from funnel_esm.metrics import summarize_metrics


def input_info(path, frame):
    records = [
        (sha256_sequence(row.Sequence), [int(x) for x in row.ec1])
        for row in frame.itertuples()
    ]
    return dict(
        csv=str(Path(path).resolve()),
        csv_sha256=sha256(path),
        n_examples=len(frame),
        ec1_truth_sha256=digest(sorted(records)),
        ordered_input_sha256=digest(records),
    )


def sha256_sequence(sequence):
    import hashlib

    return hashlib.sha256(sequence.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models_dir", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csvs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_residues", type=int, default=1022)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--precision", choices=("float32", "bf16", "fp16"), default="float32"
    )
    p.add_argument("--attention_backend", choices=("eager", "sdpa"), default="eager")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--no_energy",
        action="store_true",
        help="CPU smoke checks only: GPU energy is null",
    )
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("Inference uses one GPU/process, not distributed training")
    if (
        not 1 <= args.max_residues <= 1022
        or args.batch_size < 1
        or args.repeats < 1
        or args.warmup < 0
    ):
        p.error("Invalid length/batch/repeats/warmup")
    if args.device == "cpu" and args.precision == "fp16":
        p.error("CPU fp16 is unsupported")
    args.progress = False
    output = new_output(args.output_dir)
    train = read_care(args.train_csv)
    tests = {}
    for path in args.test_csvs:
        name = Path(path).stem
        if name in tests:
            raise ValueError("Test names must be unique")
        frame = read_care(path)
        check_disjoint(train, frame)
        tests[name] = (frame, input_info(path, frame))
    paths = sorted(Path(args.models_dir).resolve().rglob("run_result.json"))
    paths = [path for path in paths if (path.parent / "final_model").is_dir()]
    if not paths:
        raise ValueError("No completed final_model checkpoints found")
    meter = Meter(args.device, args.no_energy)
    config = {
        key: getattr(args, key)
        for key in (
            "batch_size",
            "max_residues",
            "precision",
            "attention_backend",
            "warmup",
            "repeats",
            "progress",
        )
    }
    config.update(length_policy="middle", ec1_policy="all")
    runs = []
    seen = set()
    resources = []
    for path in paths:
        trained = json.loads(path.read_text())
        architecture = trained["architecture"]
        seed = trained["seed"]
        model, tokenizer, checkpoint = load_saved_classifier(
            path.parent, CARE_LABELS, meter.device, args.attention_backend
        )
        cfg = model.config
        saved_architecture = getattr(cfg, "funnel_architecture", None)
        if saved_architecture is not None and saved_architecture != architecture:
            raise ValueError("Checkpoint and run_result.json architectures differ")
        if "pool_after_layers" in trained and trained["pool_after_layers"] != getattr(
            cfg, "funnel_pool_after_layers", []
        ):
            raise ValueError("Checkpoint and run_result.json pooling layers differ")
        tag = f"ESM-L{cfg.num_hidden_layers}-H{cfg.hidden_size}-{cfg.position_embedding_type}"
        model_name = f"FunnelESM/{tag}/{architecture}"
        key = (model_name, seed)
        if key in seen:
            raise ValueError("Duplicate model/seed; select one campaign")
        seen.add(key)
        destination = output / tag / architecture / f"seed_{seed}"
        destination.mkdir(parents=True)
        weights = {
            f.name: sha256(f)
            for f in checkpoint.iterdir()
            if f.suffix == ".safetensors"
            or f.name == "pytorch_model.bin"
            or (f.name.startswith("pytorch_model-") and f.suffix == ".bin")
        }
        result = dict(
            schema=3,
            measurement_protocol=PROTOCOL,
            model=model_name,
            training_seed=seed,
            checkpoint_id=weights,
            protocol=dict(
                config=config,
                runtime=runtime(meter.device, args.precision),
                backbone_tag=tag,
                implementation_sha256={
                    "measurement.py": sha256(
                        Path(__file__).resolve().parents[1]
                        / "src/funnel_esm/inference.py"
                    )
                },
                original_training_result=trained,
            ),
            test_metrics={},
            resources={"test_by_split": {}},
        )
        predictor = ProteinPredictor(model, tokenizer, args, meter.device)
        for name, (frame, info) in tests.items():
            truth = np.asarray(
                [[int(str(i) in row) for i in range(1, 8)] for row in frame.ec1]
            )

            def score(probabilities):
                top = np.argmax(probabilities, axis=1)
                return {
                    "top1_ec1_accuracy": float(truth[np.arange(len(truth)), top].mean())
                }

            metrics, probabilities, resource = run_passes(
                predictor, frame.Sequence.tolist(), score, args, meter, name
            )
            result["test_metrics"][name] = dict(
                n_examples=len(frame), metrics=metrics, dataset=info
            )
            result["resources"]["test_by_split"][name] = resource
            predictions = np.argmax(probabilities, axis=1) + 1
            pd.DataFrame(
                dict(
                    protein_id=frame.protein_id,
                    predicted_ec1=predictions,
                    true_ec1=["|".join(labels) for labels in frame.ec1],
                )
            ).to_csv(destination / f"{name}_predictions.csv", index=False)
        result["resources"]["benchmark_total"] = combine_resources(
            list(result["resources"]["test_by_split"].values())
        )
        save_json(destination / "evaluation_result.json", result)
        resources.append(result["resources"]["benchmark_total"])
        runs.append(result)
        del model, predictor
        gc.collect()
        torch.cuda.empty_cache()
    summary = {}
    for name in sorted({r["model"] for r in runs}):
        selected = [r for r in runs if r["model"] == name]
        summary[name] = {}
        for split in tests:
            rows = [
                dict(
                    r["test_metrics"][split]["metrics"],
                    **{
                        k: r["resources"]["test_by_split"][split][k]
                        for k in (
                            "time_s",
                            "gpu_energy_j",
                            "joules_per_protein",
                            "peak_gpu_memory_gb",
                            "proteins_per_second",
                        )
                    },
                )
                for r in selected
            ]
            summary[name][split] = summarize_metrics(rows)
    save_json(
        output / "inference_results.json",
        dict(
            config=config,
            runs=runs,
            summary=summary,
            benchmark_resources=combine_resources(resources),
            measurement_protocol=PROTOCOL,
        ),
    )
    print(f"Results: {output / 'inference_results.json'}", flush=True)


if __name__ == "__main__":
    main()
