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
    needs_grad = False

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        scores = W.abs()
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class GradientStrategy(ScoringStrategy):
    """|W| x |grad| — Requires gradients."""
    needs_grad = True

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        grad = weights.get_gradient(layer_name)
        if grad is None:
            raise ValueError(f"No gradient available for {layer_name}")
        scores = W.abs() * grad.abs()
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class WandaStrategy(ScoringStrategy):
    """|W| x ||X||_2 — Forward only, state of the art.
    
    Receives concatenated activations from all batches.
    Computes score = |W| x ||X||_2 once (like legacy).
    """
    needs_grad = False

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        # Get concatenated activations from all batches
        X = getattr(activations, 'get_input_concatenated',
                    activations.get_input_activations)(layer_name)
        if X is None:
            raise ValueError(f"No activations for {layer_name}")
        # ||X||_2 per input feature (like legacy)
        X_norm = X.norm(p=2, dim=0)
        scores = W.abs() * X_norm.unsqueeze(0)
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


class GPSStrategy(ScoringStrategy):
    """GPS Local — Direction x Selectivity x Distortion.
    
    Uses concatenated activations from ALL batches (like legacy GPS)
    to compute variance, mean, and distortion correctly.
    """
    needs_grad = False

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        
        # Use concatenated activations (all batches) like legacy GPS
        X_in = getattr(activations, 'get_input_concatenated', 
                       activations.get_input_activations)(layer_name)
        X_out = getattr(activations, 'get_output_concatenated',
                        activations.get_output_activations)(layer_name)
        
        if X_in is None or X_out is None:
            raise ValueError(f"No activations for {layer_name}")

        W_abs = W.abs()
        direction = W_abs.max(dim=1).values / (W_abs.mean(dim=1) + 1e-8)

        acts = X_in @ W.T
        selectivity = acts.var(dim=0) / (acts.mean(dim=0) + 1e-8)

        bias = weights.get_bias(layer_name)
        contrib = (X_in @ W.T) + (bias if bias is not None else 0)
        contrib = contrib.relu()
        distortion = contrib.norm(dim=0) / (X_out.norm() + 1e-8)

        scores = direction * selectivity * distortion
        scores = scores.clamp(min=0)
        scores = scores.unsqueeze(1).expand_as(W)
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


# Built-in strategies (codées en dur, rapides)
BUILTIN_STRATEGIES = {
    "magnitude": MagnitudeStrategy,
    "gradient": GradientStrategy,
    "wanda": WandaStrategy,
    "gps": GPSStrategy,
}

# APL strategies (utilisent apl-pruning)
from domain.scoring.apl_strategies import APLFormulaStrategy, APL_FORMULAS

APL_STRATEGIES = {
    name: lambda f=formula: APLFormulaStrategy(f)
    for name, formula in APL_FORMULAS.items()
    if name not in BUILTIN_STRATEGIES
}

# Registry complet
STRATEGY_REGISTRY = {**BUILTIN_STRATEGIES, **APL_STRATEGIES}


def get_strategy(name: str, apl_bridge=None) -> ScoringStrategy:
    """Scoring strategy factory.
    
    Args:
        name: Strategy name (e.g., 'wanda', 'gps_cube')
        apl_bridge: APLScoringPort instance (required for APL strategies)
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    strategy = STRATEGY_REGISTRY[name]()
    if hasattr(strategy, 'set_bridge') and apl_bridge is not None:
        strategy.set_bridge(apl_bridge)
    return strategy
