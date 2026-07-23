"""Metrics used by Hugging Face Trainer."""

import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(eval_prediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "micro_f1": f1_score(
            labels, predictions, average="micro", zero_division=0
        ),
    }
