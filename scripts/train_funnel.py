#!/usr/bin/env python3
"""Train ESM-2 with stride-2 masked-mean inter-layer reductions."""

from transformers import AutoConfig

from funnel_esm.data import DEEPLOC_LABELS
from funnel_esm.modeling_funnel_esm import (
    FunnelEsmForSequenceClassification,
)
from train_baseline import ROPE_KEY_MAPPING, build_parser, run


def resolve_pool_after_layers(args, num_hidden_layers):
    """Resolve the old single-layer option and the current list option."""
    if args.pool_after_layers is not None:
        pool_after_layers = args.pool_after_layers
    elif args.pool_after_layer is not None:
        pool_after_layers = [args.pool_after_layer]
    else:
        pool_after_layers = [num_hidden_layers // 2]

    if pool_after_layers != sorted(pool_after_layers):
        raise ValueError("--pool_after_layers must be sorted")
    if len(pool_after_layers) != len(set(pool_after_layers)):
        raise ValueError("--pool_after_layers cannot contain duplicates")
    if any(
        layer < 1 or layer >= num_hidden_layers
        for layer in pool_after_layers
    ):
        raise ValueError(
            "Pooling layers must be between 1 and "
            "num_hidden_layers - 1"
        )

    return pool_after_layers


def parse_args():
    parser = build_parser()
    parser.description = (
        "Train ESM-2 with stride-2 masked-mean inter-layer reductions."
    )
    parser.set_defaults(output_dir="outputs/funnel_mean")
    parser.add_argument(
        "--pool_after_layer",
        type=int,
        default=None,
        help=(
            "Apply one pooling operation after this 1-based layer. "
            "Kept for compatibility."
        ),
    )
    parser.add_argument(
        "--pool_after_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sorted 1-based layers after which pooling is applied, "
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


def load_funnel_model(args):
    config = AutoConfig.from_pretrained(args.model_name)
    pool_after_layers = resolve_pool_after_layers(
        args,
        config.num_hidden_layers,
    )

    config.funnel_pool_after_layers = pool_after_layers
    config.funnel_pool_stride = 2
    config.funnel_pooling_type = "pairwise_masked_mean"
    config.num_labels = len(DEEPLOC_LABELS)
    config.problem_type = "multi_label_classification"
    config.deeploc_focal_gamma = args.focal_gamma
    config.deeploc_gaussian_kernel_size = 5
    config.deeploc_gaussian_sigma = 2.0
    config.id2label = dict(enumerate(DEEPLOC_LABELS))
    config.label2id = {
        label: index for index, label in enumerate(DEEPLOC_LABELS)
    }

    return FunnelEsmForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        key_mapping=ROPE_KEY_MAPPING,
    )


def main():
    run(parse_args(), model_loader=load_funnel_model)


if __name__ == "__main__":
    main()
