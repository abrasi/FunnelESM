#!/usr/bin/env python3
"""Train mean-pool ESM-2 on DeepLoc or CARE, with explicit pooling layers."""

import argparse

from funnel_esm.care import CARE_LABELS
from funnel_esm.deeploc import DEEPLOC_LABELS
from funnel_esm.modeling_classification import load_pretrained_classifier
from funnel_esm.modeling_funnel_esm import FunnelEsmForSequenceClassification


def parse_training_args(description, architecture, argv=None):
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument("--task", choices=("deeploc", "care"), default="deeploc")
    task, _ = task_parser.parse_known_args(argv)
    if task.task == "care":
        import train_care_baseline as training
    else:
        import train_deeploc_baseline as training
    parser = training.build_parser()
    parser.description = description
    parser.add_argument("--task", choices=("deeploc", "care"), default=task.task)
    parser.add_argument(
        "--pool_after_layers",
        type=int,
        nargs="+",
        required=True,
        help="Sorted 1-based pooling layers, e.g. 11 22 for a 33-layer ESM-2",
    )
    if task.task == "care":
        parser.set_defaults(architecture=architecture)
    return parser.parse_args(argv), training.run


def load_funnel_model(args):
    labels = CARE_LABELS if args.task == "care" else DEEPLOC_LABELS
    return load_pretrained_classifier(
        FunnelEsmForSequenceClassification,
        args.model_name,
        labels,
        args.focal_gamma,
        args.pool_after_layers,
        "pairwise_masked_mean",
    )


def main():
    args, run = parse_training_args(__doc__, "mean_pool")
    run(args, model_loader=load_funnel_model)


if __name__ == "__main__":
    main()
