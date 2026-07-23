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
    max_residues=1024,
):
    """Return tokenized train and validation datasets."""

    dataset = load_dataset("csv", data_files=csv_path, split="train")
    dataset = dataset.filter(
        lambda example: len(example["Sequence"]) <= max_residues
    )

    train_dataset = dataset.filter(
        lambda example: example["Partition"] != validation_partition
    )
    validation_dataset = dataset.filter(
        lambda example: example["Partition"] == validation_partition
    )

    def tokenize(batch):
        encoded = tokenizer(batch["Sequence"], truncation=False)
        encoded["labels"] = [
            [float(batch[label][index]) for label in DEEPLOC_LABELS]
            for index in range(len(batch["Sequence"]))
        ]
        return encoded

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
        remove_columns=validation_dataset.column_names,
    )

    return DatasetDict(
        train=train_dataset,
        validation=validation_dataset,
    )
