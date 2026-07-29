"""Load and tokenize the DeepLoc 2.0 dataset."""

from datasets import DatasetDict, load_dataset


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
):
    """Return tokenized train and validation datasets.

    Sequences longer than ``max_residues`` keep their N- and C-terminal
    regions and lose only the middle of the sequence. The five official
    DeepLoc partitions are preserved without recalculating the folds.
    """

    dataset = load_dataset("csv", data_files=csv_path, split="train")

    train_dataset = dataset.filter(
        lambda example: example["Partition"] != validation_partition
    )
    validation_dataset = dataset.filter(
        lambda example: example["Partition"] == validation_partition
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
        return encoded

    datasets = DatasetDict(
        train=train_dataset,
        validation=validation_dataset,
    )
    return datasets.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
