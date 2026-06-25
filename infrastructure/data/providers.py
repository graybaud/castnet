"""Data providers — Adapters for datasets."""

from typing import Iterator
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from domain.scoring.ports import BatchProvider


class WikiTextProvider(BatchProvider):
    """Provides batches from WikiText-2."""

    def __init__(self, tokenizer_name: str, max_len: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._max_len = max_len
        self._dataset = self._load()

    def _load(self):
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1",
            split="train", streaming=False  # ← NO STREAMING
        )
        return dataset.filter(lambda x: len(x["text"].strip()) > 10)

    def get_batches(self, num_batches: int, batch_size: int = 1) -> Iterator[torch.Tensor]:
        count = 0
        for example in self._dataset:
            if count >= num_batches:
                break
            ids = self.tokenizer.encode(
                example["text"],
                truncation=True,
                max_length=self._max_len,
                return_tensors="pt",
            )
            if ids.size(1) > 1:
                yield ids
                count += 1
