#!/usr/bin/env python3
"""Train pool-query-only ESM-2 on DeepLoc or CARE, with explicit pooling layers."""

from train_funnel import parse_training_args

from funnel_esm.care import CARE_LABELS
from funnel_esm.deeploc import DEEPLOC_LABELS
from funnel_esm.modeling_classification import load_pretrained_classifier
from funnel_esm.modeling_funnel_esm import FunnelEsmPoolQueryForSequenceClassification


def load_pool_query_model(args):
    labels = CARE_LABELS if args.task == "care" else DEEPLOC_LABELS
    return load_pretrained_classifier(
        FunnelEsmPoolQueryForSequenceClassification,
        args.model_name,
        labels,
        args.focal_gamma,
        args.pool_after_layers,
        "pool_query_only_mean",
    )


def main():
    args, run = parse_training_args(__doc__, "pool_query")
    run(args, model_loader=load_pool_query_model)


if __name__ == "__main__":
    main()
