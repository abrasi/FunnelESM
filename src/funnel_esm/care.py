"""CARE Task 1 data for direct EC level-1 multilabel classification."""

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

from funnel_esm.data import truncate_protein_middle

CARE_LABELS = tuple(f"EC{i}" for i in range(1, 8))


def load_care(train_csv, tokenizer, test_csv=None, validation_size=0.1,
              validation_seed=0, max_residues=1022, cluster_metadata_csv=None):
    """Keep all EC1 annotations per unique sequence; never subsample clusters."""
    frame = pd.read_csv(train_csv, dtype={"EC number": str})
    if not {"Sequence", "EC number"} <= set(frame.columns):
        raise ValueError("CARE training CSV requires Sequence and EC number")
    frame = frame.dropna(subset=["Sequence", "EC number"]).copy()
    frame["ec1"] = frame["EC number"].str.extract(r"^([1-7])(?:\.|$)")[0]
    frame = frame.dropna(subset=["ec1"])
    if frame.empty:
        raise ValueError("No valid EC1 annotations found")
    grouped = frame.groupby("Sequence", sort=False)["ec1"].agg(lambda x: sorted(set(x)))
    proteins = pd.DataFrame({"Sequence": grouped.index, "ec1": grouped.values})
    if cluster_metadata_csv:
        meta = pd.read_csv(cluster_metadata_csv, usecols=["Sequence", "clusterRes50"])
        per_sequence = meta.drop_duplicates()
        if per_sequence.groupby("Sequence")["clusterRes50"].nunique().max() > 1:
            raise ValueError("Inconsistent clusterRes50 assignments")
        clusters = per_sequence.drop_duplicates("Sequence").set_index("Sequence")["clusterRes50"]
        proteins["group"] = proteins["Sequence"].map(clusters)
        if proteins["group"].isna().any():
            raise ValueError("Cluster metadata missing some CARE training sequences")
    else:
        proteins["group"] = proteins["Sequence"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=validation_seed)
    train_idx, val_idx = next(splitter.split(proteins, groups=proteins["group"]))

    def to_dataset(rows):
        entries = []
        for sequence, ec1 in zip(rows["Sequence"], rows["ec1"]):
            entries.append({"Sequence": sequence, "labels": [float(str(i) in ec1) for i in range(1, 8)]})
        return Dataset.from_list(entries)

    parts = {"train": to_dataset(proteins.iloc[train_idx]),
             "validation": to_dataset(proteins.iloc[val_idx])}
    if test_csv:
        test = pd.read_csv(test_csv, dtype={"EC number": str})
        if not {"Sequence", "EC number"} <= set(test.columns):
            raise ValueError("CARE test CSV requires Sequence and EC number")
        test = test.dropna(subset=["Sequence", "EC number"]).copy()
        test["ec1"] = test["EC number"].str.extract(r"^([1-7])(?:\.|$)")[0]
        test = test.dropna(subset=["ec1"])
        test_groups = test.groupby("Sequence", sort=False)["ec1"].agg(lambda x: sorted(set(x)))
        test_proteins = pd.DataFrame({"Sequence": test_groups.index, "ec1": test_groups.values})
        if set(test_proteins["Sequence"]) & set(proteins["Sequence"]):
            raise ValueError("CARE train/test have overlapping sequences")
        parts["test"] = to_dataset(test_proteins)

    def tokenize(batch):
        encoded = tokenizer([truncate_protein_middle(s, max_residues) for s in batch["Sequence"]], truncation=False)
        encoded["labels"] = batch["labels"]
        return encoded

    return DatasetDict(parts).map(tokenize, batched=True, remove_columns=["Sequence"])
