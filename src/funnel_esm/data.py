"""Load and tokenize the DeepLoc 2.0 dataset."""

from datasets import DatasetDict, load_dataset
from sklearn.model_selection import ShuffleSplit


DEEPLOC_LABELS = (
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
)


def load_deeploc(
    csv_path,
    tokenizer,
    validation_partition=0,
    max_residues=1022,
    use_internal_validation=False,
    internal_validation_size=2048,
    internal_validation_seed=0,
):
    """Return tokenized train and validation datasets.

    Sequences longer than ``max_residues`` keep their N- and C-terminal
    regions and lose only the middle of the sequence. The five official
    DeepLoc partitions are preserved without recalculating the folds.

    Final experiments reserve ``internal_validation_size`` examples from
    the four outer-training partitions for checkpoint selection. The held
    official partition remains untouched until final evaluation, following
    the released DeepLoc 2.0 training protocol. Convergence pilots leave
    this option disabled and retain their original behavior.
    """

    dataset = load_dataset("csv", data_files=csv_path, split="train")
    if use_internal_validation:
        dataset = dataset.map(
            lambda _example, index: {"_source_row": index},
            with_indices=True,
        )

    outer_train_dataset = dataset.filter(
        lambda example: example["Partition"] != validation_partition
    )
    outer_validation_dataset = dataset.filter(
        lambda example: example["Partition"] == validation_partition
    )

    if use_internal_validation:
        if len(outer_train_dataset) <= internal_validation_size:
            raise ValueError(
                "The outer training set must contain more examples than "
                "the internal validation set."
            )
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=internal_validation_size,
            random_state=internal_validation_seed,
        )
        train_indices, validation_indices = next(
            splitter.split(range(len(outer_train_dataset)))
        )
        train_dataset = outer_train_dataset.select(train_indices)
        validation_dataset = outer_train_dataset.select(
            validation_indices
        )
    else:
        train_dataset = outer_train_dataset
        validation_dataset = outer_validation_dataset

    identifier_candidates = (
        "ACC",
        "Entry",
        "Accession",
        "Protein_ID",
        "protein_id",
        "ID",
    )
    identifier_column = None
    if use_internal_validation:
        identifier_column = next(
            (
                column
                for column in identifier_candidates
                if column in dataset.column_names
            ),
            None,
        )

    def tokenize(batch):
        left_residues = max_residues // 2
        right_residues = max_residues - left_residues
        sequences = [
            (
                sequence
                if len(sequence) <= max_residues
                else (
                    sequence[:left_residues]
                    + sequence[-right_residues:]
                )
            )
            for sequence in batch["Sequence"]
        ]

        encoded = tokenizer(sequences, truncation=False)
        encoded["labels"] = [
            [float(batch[label][index]) for label in DEEPLOC_LABELS]
            for index in range(len(batch["Sequence"]))
        ]
        if use_internal_validation:
            if identifier_column is None:
                encoded["protein_id"] = [
                    f"row_{source_row}"
                    for source_row in batch["_source_row"]
                ]
            else:
                encoded["protein_id"] = [
                    str(identifier)
                    for identifier in batch[identifier_column]
                ]
        return encoded

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
    }
    if use_internal_validation:
        datasets["threshold_calibration"] = outer_train_dataset
        datasets["outer_validation"] = outer_validation_dataset

    return DatasetDict(datasets).map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
