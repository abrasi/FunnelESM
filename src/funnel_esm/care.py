"""CARE Task 1 data loading for direct EC level-1 multilabel learning."""

from __future__ import annotations

import re

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

from funnel_esm.deeploc import truncate_protein_middle


CARE_LABELS = tuple(f"EC{i}" for i in range(1, 8))


def _normalize_sequence(value):
    if not isinstance(value, str):
        return None
    sequence = re.sub(r"\s+", "", value).upper()
    return sequence or None


def _read_protein_ec_csv(csv_path):
    frame = pd.read_csv(csv_path, dtype={"EC number": str})
    required = {"Sequence", "EC number"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
        )
    frame = frame.dropna(subset=["Sequence", "EC number"]).copy()
    frame["Sequence"] = frame["Sequence"].map(_normalize_sequence)
    frame["ec1"] = frame["EC number"].astype(str).str.extract(
        r"^\s*([1-7])(?:\.|\s|$)"
    )[0]
    frame = frame.dropna(subset=["Sequence", "ec1"])
    if frame.empty:
        raise ValueError(f"No valid sequence/EC1 records found in {csv_path}")

    records = []
    for sequence, rows in frame.groupby("Sequence", sort=False):
        entries = (
            rows["Entry"].dropna().astype(str).unique()
            if "Entry" in rows.columns
            else []
        )
        records.append(
            {
                "Sequence": sequence,
                "ec1": sorted(set(rows["ec1"].astype(str))),
                "protein_id": "|".join(entries)
                if len(entries)
                else f"sequence_{len(records)}",
            }
        )
    return pd.DataFrame.from_records(records)


def _cluster_groups(proteins, cluster_metadata_csv):
    if cluster_metadata_csv is None:
        return proteins["Sequence"]

    metadata = pd.read_csv(
        cluster_metadata_csv,
        usecols=["Sequence", "clusterRes50"],
    )
    metadata["Sequence"] = metadata["Sequence"].map(_normalize_sequence)
    metadata = metadata.dropna(subset=["Sequence", "clusterRes50"])
    grouped = metadata.groupby("Sequence", sort=False)["clusterRes50"]
    inconsistent = grouped.nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if not inconsistent.empty:
        raise ValueError("A sequence maps to multiple clusterRes50 values")
    cluster_map = grouped.first()
    groups = proteins["Sequence"].map(cluster_map)
    if groups.isna().any():
        missing_count = int(groups.isna().sum())
        raise ValueError(
            f"Cluster metadata has no clusterRes50 for {missing_count} "
            "training sequences"
        )
    return groups.astype(str)


def _to_dataset(proteins):
    rows = []
    for record in proteins.itertuples(index=False):
        rows.append(
            {
                "Sequence": record.Sequence,
                "protein_id": record.protein_id,
                "labels": [
                    float(str(index) in record.ec1)
                    for index in range(1, len(CARE_LABELS) + 1)
                ],
            }
        )
    return Dataset.from_list(rows)


def load_care(
    train_csv,
    tokenizer,
    test_csv=None,
    validation_size=0.1,
    validation_seed=0,
    max_residues=1022,
    cluster_metadata_csv=None,
):
    """Return CARE train/validation, calibration, and optional official test.

    Rows with identical normalized sequences are combined and all their EC1
    labels are retained. The official CARE test is loaded only when
    ``test_csv`` is provided. In final runs, ``threshold_calibration`` is the
    complete official training set, mirroring DeepLoc's use of outer-training
    data for MCC threshold calibration.
    """
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    proteins = _read_protein_ec_csv(train_csv)
    groups = _cluster_groups(proteins, cluster_metadata_csv)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=validation_seed,
    )
    train_indices, validation_indices = next(
        splitter.split(proteins, groups=groups)
    )

    parts = {
        "train": _to_dataset(proteins.iloc[train_indices]),
        "validation": _to_dataset(proteins.iloc[validation_indices]),
        "threshold_calibration": _to_dataset(proteins),
    }
    if test_csv is not None:
        test_proteins = _read_protein_ec_csv(test_csv)
        overlap = set(proteins["Sequence"]) & set(test_proteins["Sequence"])
        if overlap:
            raise ValueError(
                f"CARE train/test contain {len(overlap)} identical sequences"
            )
        parts["test"] = _to_dataset(test_proteins)

    def tokenize(batch):
        sequences = [
            truncate_protein_middle(sequence, max_residues)
            for sequence in batch["Sequence"]
        ]
        encoded = tokenizer(sequences, truncation=False)
        encoded["labels"] = batch["labels"]
        encoded["protein_id"] = batch["protein_id"]
        return encoded

    return DatasetDict(parts).map(
        tokenize,
        batched=True,
        remove_columns=["Sequence"],
    )
