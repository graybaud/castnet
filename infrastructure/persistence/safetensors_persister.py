"""Safetensors persistence — Implements ScorePersister and MaskPersister."""

import os
import torch
from safetensors.torch import save_file, load_file
from domain.scoring.ports import ScorePersister, MaskPersister


class SafetensorsScorePersister(ScorePersister):
    """Save/load scores in Safetensors format."""

    def save(self, scores: dict[str, torch.Tensor], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(scores, path)

    def load(self, path: str) -> dict[str, torch.Tensor]:
        return load_file(path)


class SafetensorsMaskPersister(MaskPersister):
    """Save/load masks in Safetensors format."""

    def save(self, masks: dict[str, torch.Tensor], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(masks, path)

    def load(self, path: str) -> dict[str, torch.Tensor]:
        return load_file(path)
