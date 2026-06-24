"""
Scoring strategies — Pure domain logic.

Each strategy is a mathematical formula.
No dependency on infrastructure.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from domain.scoring.ports import WeightProvider, ActivationProvider


class ScoringStrategy(ABC):
    """Port — Scoring strategy interface."""

    @abstractmethod
    def calculate(
        self,
        weights: WeightProvider,
        activations: ActivationProvider,
        layer_name: str,
    ) -> Tensor:
        """Computes importance scores for a given layer."""
        ...


class MagnitudeStrategy(ScoringStrategy):
    """|W| — Naive baseline."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        scores = W.abs()
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class GradientStrategy(ScoringStrategy):
    """|W| x |grad| — Requires gradients."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        grad = weights.get_gradient(layer_name)
        if grad is None:
            raise ValueError(f"No gradient available for {layer_name}")
        scores = W.abs() * grad.abs()
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class WandaStrategy(ScoringStrategy):
    """|W| x ||X||_2 — Forward only, state of the art."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        X = activations.get_input_activations(layer_name)
        X_norm = X.norm(p=2, dim=0)  # [d_in]
        scores = W.abs() * X_norm.unsqueeze(0)  # [d_out, d_in]
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class GPSStrategy(ScoringStrategy):
    """GPS Local — Direction x Selectivity x Distortion."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        X_in = activations.get_input_activations(layer_name)
        X_out = activations.get_output_activations(layer_name)

        # Direction: max/mean per neuron
        W_abs = W.abs()
        direction = W_abs.max(dim=1).values / (W_abs.mean(dim=1) + 1e-8)

        # Selectivity: var/mean of projected activations
        acts = X_in @ W.T
        selectivity = acts.var(dim=0) / (acts.mean(dim=0) + 1e-8)

        # Distortion: contribution norm / output norm
        bias = weights.get_bias(layer_name)
        contrib = (X_in @ W.T) + (bias if bias is not None else 0)
        contrib = contrib.relu()
        distortion = contrib.norm(dim=0) / (X_out.norm() + 1e-8)

        scores = direction * selectivity * distortion
        scores = scores.clamp(min=0)
        scores = scores.unsqueeze(1).expand_as(W)
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


# Registry
STRATEGY_REGISTRY = {
    "magnitude": MagnitudeStrategy,
    "gradient": GradientStrategy,
    "wanda": WandaStrategy,
    "gps": GPSStrategy,
}


def get_strategy(name: str) -> ScoringStrategy:
    """Scoring strategy factory."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]()
