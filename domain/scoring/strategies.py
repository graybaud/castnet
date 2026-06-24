"""
Strategies de scoring — Metier pur.

Chaque strategie est une formule mathematique.
Aucune dependance a l'infrastructure.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from domain.scoring.ports import WeightProvider, ActivationProvider


class ScoringStrategy(ABC):
    """Port — Strategie de scoring."""

    @abstractmethod
    def calculate(
        self,
        weights: WeightProvider,
        activations: ActivationProvider,
        layer_name: str,
    ) -> Tensor:
        """Calcule les scores d'importance pour une couche."""
        ...


class MagnitudeStrategy(ScoringStrategy):
    """|W| — Baseline naive."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        scores = W.abs()
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class GradientStrategy(ScoringStrategy):
    """|W| x |grad| — Necessite les gradients."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        grad = weights.get_gradient(layer_name)
        if grad is None:
            raise ValueError(f"Pas de gradient pour {layer_name}")
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
    """GPS Local — Direction x Selectivite x Distorsion."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        X_in = activations.get_input_activations(layer_name)
        X_out = activations.get_output_activations(layer_name)

        # Direction : max/mean par neurone
        W_abs = W.abs()
        direction = W_abs.max(dim=1).values / (W_abs.mean(dim=1) + 1e-8)

        # Selectivite : var/mean des activations projetees
        acts = X_in @ W.T
        selectivity = acts.var(dim=0) / (acts.mean(dim=0) + 1e-8)

        # Distorsion : norme contribution / norme sortie
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
    """Fabrique de strategie de scoring."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategie inconnue '{name}'. Disponibles: {list(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]()
