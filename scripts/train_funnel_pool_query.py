#!/usr/bin/env python3
"""Train ESM-2 with one stride-2 pool-query-only transition."""

from transformers import AutoConfig

from funnel_esm.data import DEEPLOC_LABELS
from funnel_esm.modeling_funnel_esm import (
    FunnelEsmPoolQueryForSequenceClassification,
)
from train_baseline import build_parser, run


ROPE_KEY_MAPPING = {
    "esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq":
        "esm.rotary_embeddings.inv_freq",
}


def parse_args():
    parser = build_parser()
    parser.description = (
        "Train ESM-2 with one stride-2 pool-query-only transition."
    )
    parser.set_defaults(output_dir="outputs/funnel_pool_query")
    parser.add_argument(
        "--pool_after_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "1-based layer after which Q is pooled. "
            "Defaults to the middle layer."
        ),
    )

    return parser.parse_args()


def load_pool_query_model(args):
    config = AutoConfig.from_pretrained(args.model_name)
    pool_after_layers = (
        args.pool_after_layers
        if args.pool_after_layers is not None
        else [config.num_hidden_layers // 2]
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
