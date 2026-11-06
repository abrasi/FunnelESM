"""CARE EC1: merge annotations, split training data, and tokenize sequences."""

import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

from funnel_esm.deeploc import truncate_protein_middle

CARE_LABELS = tuple(f"EC{i}" for i in range(1, 8))
VALID_EC = re.compile(r"[1-7]\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|n\d+|-)")


def normalize_sequence(value):
    sequence = re.sub(r"\s+", "", str(value)).upper()
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZUO]+", sequence):
        raise ValueError(f"Invalid protein sequence: {sequence[:30]!r}")
    return sequence


def read_care(path):
    """One example per sequence, with the union of all its EC1 annotations."""
    frame = pd.read_csv(path, dtype=str, sep=None, engine="python", keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    missing = {"Entry", "Sequence", "EC number"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: empty dataset")
    frame["Sequence"] = frame["Sequence"].map(normalize_sequence)
    frame["Entry"] = frame["Entry"].str.strip()
    if (frame["Entry"] == "").any() or (frame.groupby("Entry")["Sequence"].nunique() > 1).any():
        raise ValueError(f"{path}: empty Entry or an Entry has multiple sequences")

    labels = []
    for line, cell in enumerate(frame["EC number"], 2):
        annotations = [ec.strip() for ec in cell.split(";")]
        if any(not VALID_EC.fullmatch(ec) for ec in annotations):
            raise ValueError(f"{path}:{line}: invalid EC annotation {cell!r}")
        labels.append({ec[0] for ec in annotations})
    frame["ec1"] = labels

    proteins = []
    for sequence, rows in frame.groupby("Sequence", sort=False):
        proteins.append(
            {
                "Sequence": sequence,
                "protein_id": "|".join(rows["Entry"].unique()),
                "ec1": sorted(set().union(*rows["ec1"])),
            }
        )
    return pd.DataFrame(proteins)


def cluster_groups(proteins, metadata_path):
    if metadata_path is None:
        return proteins["Sequence"]
    metadata = pd.read_csv(
        metadata_path, dtype=str, sep=None, engine="python", keep_default_na=False
    )
    metadata["Sequence"] = metadata["Sequence"].map(normalize_sequence)
    if (metadata["clusterRes50"].str.strip() == "").any():
        raise ValueError("Cluster metadata contains empty clusterRes50 values")
    grouped = metadata.groupby("Sequence")["clusterRes50"]
    if (grouped.nunique() > 1).any():
        raise ValueError("A sequence maps to multiple clusterRes50 values")
    groups = proteins["Sequence"].map(grouped.first())
    if groups.isna().any():
        raise ValueError("Cluster metadata must cover every training sequence")
    return groups


def check_disjoint(train, test):
    if set(train["Sequence"]) & set(test["Sequence"]):
        raise ValueError("CARE train/test contain identical sequences")
    train_entries = {entry for ids in train["protein_id"] for entry in ids.split("|")}
    test_entries = {entry for ids in test["protein_id"] for entry in ids.split("|")}
    if train_entries & test_entries:
        raise ValueError("CARE train/test share Entry identifiers")


def tokenize_care(proteins, tokenizer, max_residues):
    dataset = Dataset.from_dict(
        {
            "Sequence": proteins["Sequence"].tolist(),
            "protein_id": proteins["protein_id"].tolist(),
            "labels": [
                [float(str(i) in labels) for i in range(1, 8)] for labels in proteins["ec1"]
            ],
        }
    )

    def tokenize(batch):
        sequences = [truncate_protein_middle(s, max_residues) for s in batch["Sequence"]]
        return tokenizer(sequences, truncation=False)

    return dataset.map(tokenize, batched=True, remove_columns=["Sequence"])


def load_care(
    train_csv, tokenizer, *, pilot=False, test_csvs=(), max_residues=1022, cluster_metadata_csv=None
):
    """Pilot: fixed 90/10 train/validation. Final: all train and official tests.

    Identical sequences are grouped before splitting. With metadata, whole
    clusterRes50 groups stay together. Neither split discards annotations.
    """
    if not 1 <= max_residues <= 1022:
        raise ValueError("max_residues must be between 1 and 1022")
    if pilot and test_csvs:
        raise ValueError("A pilot must not load official tests")
    proteins = read_care(train_csv)
    if pilot:
        groups = cluster_groups(proteins, cluster_metadata_csv)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_indices, val_indices = next(splitter.split(proteins, groups=groups))
        parts = {"train": proteins.iloc[train_indices], "validation": proteins.iloc[val_indices]}
    else:
        parts = {"train": proteins}
        for path in test_csvs:
            name = Path(path).stem
            if name in parts:
                raise ValueError(f"Duplicate or reserved test name: {name}")
            test = read_care(path)
            check_disjoint(proteins, test)
            parts[name] = test
    return DatasetDict(
        {name: tokenize_care(frame, tokenizer, max_residues) for name, frame in parts.items()}
    )
