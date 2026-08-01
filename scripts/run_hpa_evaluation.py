#!/usr/bin/env python3
"""Launch the independent HPA evaluation with one torchrun process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Launch the HPA worker on exactly one GPU, matching the final "
            "DeepLoc experiment launcher."
        )
    )
    parser.add_argument("--final_output_dir", required=True)
    parser.add_argument("--hpa_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Defaults to the value stored in the final manifest.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default=None,
        help="Defaults to the value stored in the final manifest.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        str(project_root / "scripts" / "evaluate_hpa.py"),
        "--final_output_dir",
        str(Path(args.final_output_dir).resolve()),
        "--hpa_path",
        str(Path(args.hpa_path).resolve()),
    ]
    if args.output_dir is not None:
        command.extend(
            ["--output_dir", str(Path(args.output_dir).resolve())]
        )
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.mixed_precision is not None:
        command.extend(
            ["--mixed_precision", args.mixed_precision]
        )
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
