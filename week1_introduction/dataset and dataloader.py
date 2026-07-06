import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding


class TransformerFineTuneDataset(Dataset):
    """Dataset for transformer fine-tuning on text classification or sequence regression tasks."""

    def __init__(
        self,
        examples,
        tokenizer,
        max_length=512,
        labels=None,
        text_key="text",
        text_pair_key=None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.text_key = text_key
        self.text_pair_key = text_pair_key

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if isinstance(example, dict):
            text = example.get(self.text_key)
            text_pair = example.get(self.text_pair_key) if self.text_pair_key else None
            label = example.get("label") if self.labels is None else None
        else:
            text = example
            text_pair = None
            label = None

        encoding = self.tokenizer(
            text,
            text_pair,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}

        if self.labels is not None:
            label_value = self.labels[idx]
            item["labels"] = torch.tensor(label_value, dtype=torch.long)
        elif label is not None:
            item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


def build_dataloader(
    examples,
    tokenizer,
    batch_size=8,
    shuffle=False,
    max_length=512,
    labels=None,
    text_key="text",
    text_pair_key=None,
    num_workers=0,
):
    dataset = TransformerFineTuneDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        labels=labels,
        text_key=text_key,
        text_pair_key=text_pair_key,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )


def _example_batches(tokenizer):
    examples = [
        {"text": "Hello world", "label": 0},
        {"text": "Transformers are great", "label": 1},
    ]
    return build_dataloader(examples, tokenizer, batch_size=2)


def main():
    """Run a small demo that prints batches in a readable style."""
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required to run the demo") from e

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataloader = _example_batches(tokenizer)

    for i, batch in enumerate(dataloader):
        # convert tensors to python lists for readable printing
        printable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in batch.items()}
        print(f"Batch {i}:")
        print(printable)


if __name__ == "__main__":
    main()
