"""Persistance Safetensors — Implemente ScorePersister et MaskPersister."""

import torch
from safetensors.torch import save_file, load_file
from domain.scoring.ports import ScorePersister, MaskPersister


class SafetensorsScorePersister(ScorePersister):
    """Sauvegarde/charge les scores au format Safetensors."""

    def save(self, scores: dict[str, torch.Tensor], path: str) -> None:
        save_file(scores, path)

    def load(self, path: str) -> dict[str, torch.Tensor]:
        return load_file(path)


class SafetensorsMaskPersister(MaskPersister):
    """Sauvegarde/charge les masques au format Safetensors."""

    def save(self, masks: dict[str, torch.Tensor], path: str) -> None:
        save_file(masks, path)

    def load(self, path: str) -> dict[str, torch.Tensor]:
        return load_file(path)
