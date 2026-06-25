"""WikiText-2 validation dataset loader — Infrastructure adapter."""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class WikiTextValidationDataset:
    """Loads WikiText-2 validation set as contiguous token blocks."""

    def __init__(
        self,
        tokenizer_name: str = "facebook/opt-125m",
        block_size: int = 512,
        max_tokens: int = 50000,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.block_size = block_size
        self.max_tokens = max_tokens
        self._tokens = self._load()

    def _load(self) -> torch.Tensor:
        """Load and tokenize WikiText-2 validation as one long sequence."""
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
        )
        text = " ".join(
            [x["text"] for x in dataset if len(x["text"].strip()) > 10]
        )
        return self.tokenizer.encode(text, return_tensors="pt")[0]

    def get_blocks(self, device: str = "cpu"):
        """Yield contiguous blocks of tokens."""
        tokens = self._tokens
        limit = min(len(tokens) - self.block_size, self.max_tokens)
        for i in range(0, limit, self.block_size):
            yield tokens[i : i + self.block_size].unsqueeze(0).to(device)
