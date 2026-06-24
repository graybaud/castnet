"""
Ports — Interfaces définies dans le domaine métier.

L'infrastructure DOIT implémenter ces interfaces.
Le domaine ne dépend QUE de ces abstractions.
"""

from abc import ABC, abstractmethod
from typing import Iterator
import torch
from torch import Tensor


class WeightProvider(ABC):
    """Port — Fournit les poids d'un modele, couche par couche."""

    @abstractmethod
    def get_weight(self, layer_name: str) -> Tensor:
        """Retourne la matrice de poids W [d_out, d_in]."""
        ...

    @abstractmethod
    def get_bias(self, layer_name: str) -> Tensor | None:
        """Retourne le biais [d_out], ou None."""
        ...

    @abstractmethod
    def get_gradient(self, layer_name: str) -> Tensor | None:
        """Retourne le gradient accumule [d_out, d_in], ou None."""
        ...

    @abstractmethod
    def layer_names(self) -> list[str]:
        """Liste toutes les couches eligibles au scoring."""
        ...

    @abstractmethod
    def forward_backward(self, batch: Tensor) -> float:
        """Forward + backward sur un batch. Retourne la loss."""
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Remet les gradients a zero."""
        ...


class ActivationProvider(ABC):
    """Port — Fournit les activations collectees pendant le forward."""

    @abstractmethod
    def get_input_activations(self, layer_name: str) -> Tensor:
        """Activations d'entree X pour une couche [N, d_in]."""
        ...

    @abstractmethod
    def get_output_activations(self, layer_name: str) -> Tensor:
        """Activations de sortie pour une couche [N, d_out]."""
        ...

    @abstractmethod
    def attach(self) -> None:
        """Enregistre les hooks sur toutes les couches."""
        ...

    @abstractmethod
    def detach(self) -> None:
        """Supprime tous les hooks."""
        ...


class BatchProvider(ABC):
    """Port — Fournit des batches de tokens calibres."""

    @abstractmethod
    def get_batches(self, num_batches: int, batch_size: int = 1) -> Iterator[Tensor]:
        """Itere sur des batches de tokens [B, S]."""
        ...


class ScorePersister(ABC):
    """Port — Sauvegarde et charge les scores."""

    @abstractmethod
    def save(self, scores: dict[str, Tensor], path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> dict[str, Tensor]:
        ...


class MaskPersister(ABC):
    """Port — Sauvegarde et charge les masques binaires."""

    @abstractmethod
    def save(self, masks: dict[str, Tensor], path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> dict[str, Tensor]:
        ...
