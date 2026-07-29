"""DeepLoc-style metrics and per-label threshold calibration."""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    matthews_corrcoef,
)

from funnel_esm.data import DEEPLOC_LABELS


def label_metric_name(label: str) -> str:
    """Return a stable metric suffix for one localization label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


MCC_METRIC_NAMES = tuple(
    f"mcc_{label_metric_name(label)}"
    for label in DEEPLOC_LABELS
)


def sigmoid(logits):
    """Numerically stable NumPy sigmoid."""
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1 / (1 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probabilities[~positive] = exponential / (1 + exponential)
    return probabilities


def inverse_frequency_label_weights(labels):
    """Compute normalized inverse-frequency weights from training labels."""
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2:
        raise ValueError("labels must have shape [N, num_labels]")

    counts = labels.sum(axis=0)
    if np.any(counts <= 0):
        raise ValueError("each label must occur in the training split")

    weights = 1 / counts
    return weights / weights.mean()


def optimize_mcc_thresholds(probabilities, labels):
    """Choose one exact MCC-maximizing threshold per label.

    Candidate thresholds are the distinct probabilities produced on the
    training split. Ties prefer the higher threshold.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")

    thresholds = np.empty(probabilities.shape[1], dtype=np.float64)
    for label_index in range(probabilities.shape[1]):
        scores = probabilities[:, label_index]
        targets = labels[:, label_index]
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_targets = targets[order]

        true_positives = np.cumsum(sorted_targets)
        false_positives = np.cumsum(1 - sorted_targets)
        group_ends = np.flatnonzero(
            np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
        )

        tp = true_positives[group_ends].astype(np.float64)
        fp = false_positives[group_ends].astype(np.float64)
        positives = float(sorted_targets.sum())
        negatives = float(len(sorted_targets) - positives)
        fn = positives - tp
        tn = negatives - fp

        denominator = np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = np.divide(
            tp * tn - fp * fn,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        candidate_thresholds = np.r_[
            np.nextafter(sorted_scores[0], np.inf),
            sorted_scores[group_ends],
        ]
        candidate_mcc = np.r_[0.0, mcc]
        thresholds[label_index] = candidate_thresholds[
            np.argmax(candidate_mcc)
        ]

    return thresholds


def apply_thresholds(
    probabilities,
    thresholds,
    force_at_least_one=True,
):
    """Convert probabilities to labels using DeepLoc-style thresholds."""
    probabilities = np.asarray(probabilities)
    thresholds = np.asarray(thresholds)
    if thresholds.shape != (probabilities.shape[1],):
        raise ValueError("thresholds must contain one value per label")

    predictions = (probabilities >= thresholds).astype(np.int64)
    if force_at_least_one:
        empty_rows = predictions.sum(axis=1) == 0
        if np.any(empty_rows):
            margins = probabilities[empty_rows] - thresholds
            closest_labels = np.argmax(margins, axis=1)
            predictions[
                np.flatnonzero(empty_rows),
                closest_labels,
            ] = 1
    return predictions


def multilabel_metrics(labels, probabilities, thresholds):
    """Compute the metrics reported for DeepLoc 2.0."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = apply_thresholds(
        probabilities,
        thresholds,
        force_at_least_one=True,
    )

    metrics = {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "jaccard": jaccard_score(
            labels,
            predictions,
            average="samples",
            zero_division=0,
        ),
        "accuracy": float(
            np.all(predictions == labels, axis=1).mean()
        ),
        "num_labels_predicted": float(
            predictions.sum(axis=1).mean()
        ),
    }

    per_label_mcc = []
    for label_index, metric_name in enumerate(MCC_METRIC_NAMES):
        mcc = matthews_corrcoef(
            labels[:, label_index],
            predictions[:, label_index],
        )
        metrics[metric_name] = float(mcc)
        per_label_mcc.append(mcc)
    metrics["macro_mcc"] = float(np.mean(per_label_mcc))
    return metrics


def build_compute_metrics(thresholds=None):
    """Create a Hugging Face ``compute_metrics`` callback."""
    if thresholds is None:
        thresholds = np.full(len(DEEPLOC_LABELS), 0.5)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    def compute_metrics(eval_prediction):
        probabilities = sigmoid(eval_prediction.predictions)
        return multilabel_metrics(
            eval_prediction.label_ids,
            probabilities,
            thresholds,
        )

    return compute_metrics


compute_metrics = build_compute_metrics()
