#!/usr/bin/env python3
"""Train an ESM-2 baseline on one DeepLoc 2.0 fold."""

from __future__ import annotations

import argparse

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from funnel_esm.data import DEEPLOC_LABELS, load_deeploc
from funnel_esm.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="outputs/baseline")
    parser.add_argument("--validation_partition", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--model_name",
        default="facebook/esm2_t12_35M_UR50D",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = load_deeploc(
        args.dataset_path,
        tokenizer,
        validation_partition=args.validation_partition,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(DEEPLOC_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(DEEPLOC_LABELS)),
        label2id={label: index for index, label in enumerate(DEEPLOC_LABELS)},
    )

    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
