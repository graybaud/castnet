"""Fast data provider using pre-tokenized cached dataset."""

from typing import Iterator
import torch
from domain.scoring.ports import BatchProvider


class CachedWikiTextProvider(BatchProvider):
    """Provides batches from a pre-tokenized cached file."""

    def __init__(self, cache_path: str = "/tmp/wiki_500_cached.pt"):
        self._data = torch.load(cache_path)
        self._idx = 0

    def get_batches(self, num_batches: int, batch_size: int = 1) -> Iterator[torch.Tensor]:
        for i in range(num_batches):
            idx = (self._idx + i) % len(self._data)
            yield self._data[idx].unsqueeze(0)  # [1, S]
        self._idx = (self._idx + num_batches) % len(self._data)
