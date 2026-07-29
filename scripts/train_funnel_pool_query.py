#!/usr/bin/env python3
"""Train ESM-2 with stride-2 pool-query-only transitions."""

from transformers import AutoConfig

from funnel_esm.data import DEEPLOC_LABELS
from funnel_esm.modeling_funnel_esm import (
    FunnelEsmPoolQueryForSequenceClassification,
)
from train_baseline import ROPE_KEY_MAPPING, build_parser, run
from train_funnel import resolve_pool_after_layers


def parse_args():
    parser = build_parser()
    parser.description = (
        "Train ESM-2 with stride-2 pool-query-only transitions."
    )
    parser.set_defaults(output_dir="outputs/funnel_pool_query")
    parser.add_argument(
        "--pool_after_layer",
        type=int,
        default=None,
        help=(
            "Apply one Q pooling operation after this 1-based layer. "
            "Kept for compatibility."
        ),
    )
    parser.add_argument(
        "--pool_after_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sorted 1-based layers after which Q is pooled, "
            "for example: 4 8."
        ),
    )
    args = parser.parse_args()
    if (
        args.pool_after_layer is not None
        and args.pool_after_layers is not None
    ):
        parser.error(
            "Use either --pool_after_layer or --pool_after_layers, not both."
        )
    return args


def load_pool_query_model(args):
    config = AutoConfig.from_pretrained(args.model_name)
    pool_after_layers = resolve_pool_after_layers(
        args,
        config.num_hidden_layers,
    )

    config.funnel_pool_after_layers = pool_after_layers
    config.funnel_pool_stride = 2
    config.funnel_pooling_type = "pool_query_only_mean"
    config.num_labels = len(DEEPLOC_LABELS)
    config.problem_type = "multi_label_classification"
    config.id2label = dict(enumerate(DEEPLOC_LABELS))
    config.label2id = {
        label: index for index, label in enumerate(DEEPLOC_LABELS)
    }

    return FunnelEsmPoolQueryForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        key_mapping=ROPE_KEY_MAPPING,
    )


def main():
    run(parse_args(), model_loader=load_pool_query_model)


if __name__ == "__main__":
    main()
