"""
Ports — Interfaces defined in the domain layer.

Infrastructure MUST implement these interfaces.
Domain depends ONLY on these abstractions.
"""

from abc import ABC, abstractmethod
from typing import Iterator
import torch
from torch import Tensor


class WeightProvider(ABC):
    """Port — Provides model weights, layer by layer."""

    @abstractmethod
    def get_weight(self, layer_name: str) -> Tensor:
        """Returns weight matrix W [d_out, d_in]."""
        ...

    @abstractmethod
    def get_bias(self, layer_name: str) -> Tensor | None:
        """Returns bias vector [d_out], or None."""
        ...

    @abstractmethod
    def get_gradient(self, layer_name: str) -> Tensor | None:
        """Returns accumulated gradient [d_out, d_in], or None."""
        ...

    @abstractmethod
    def layer_names(self) -> list[str]:
        """Lists all layers eligible for scoring."""
        ...

    @abstractmethod
    def forward_backward(self, batch: Tensor) -> float:
        """Forward + backward on a batch. Returns loss."""
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Resets gradients to zero."""
        ...


class ActivationProvider(ABC):
    """Port — Provides activations collected during forward pass."""

    @abstractmethod
    def get_input_activations(self, layer_name: str) -> Tensor:
        """Input activations X for a layer [N, d_in]."""
        ...

    @abstractmethod
    def get_output_activations(self, layer_name: str) -> Tensor:
        """Output activations for a layer [N, d_out]."""
        ...

    @abstractmethod
    def attach(self) -> None:
        """Registers hooks on all target layers."""
        ...

    @abstractmethod
    def detach(self) -> None:
        """Removes all hooks."""
        ...


class BatchProvider(ABC):
    """Port — Provides calibrated token batches."""

    @abstractmethod
    def get_batches(self, num_batches: int, batch_size: int = 1) -> Iterator[Tensor]:
        """Iterates over token batches [B, S]."""
        ...


class ScorePersister(ABC):
    """Port — Saves and loads scores."""

    @abstractmethod
    def save(self, scores: dict[str, Tensor], path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> dict[str, Tensor]:
        ...


class MaskPersister(ABC):
    """Port — Saves and loads binary masks."""

    @abstractmethod
    def save(self, masks: dict[str, Tensor], path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> dict[str, Tensor]:
        ...

class APLScoringPort(ABC):
    """Port — Evaluates APL formulas against weight/activation providers."""

    @abstractmethod
    def calculate(self, weights: WeightProvider, activations: ActivationProvider,
                  layer_name: str) -> "torch.Tensor":
        """Compute scores for a layer using an APL formula."""
        ...

