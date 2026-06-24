"""APL-based scoring strategies — delegates to apl-pruning formulas."""

import torch
from torch import Tensor
from domain.scoring.strategies import ScoringStrategy
from infrastructure.scoring.apl_bridge import APLScoringBridge


class APLFormulaStrategy(ScoringStrategy):
    """Generic strategy driven by an APL formula string.

    Usage:
        strategy = APLFormulaStrategy("|W| x mean(|act|)")
        scores = strategy.calculate(weights, activations, "layer1")
    """

    def __init__(self, formula: str):
        self.bridge = APLScoringBridge(formula)

    def calculate(self, weights, activations, layer_name) -> Tensor:
        scores = self.bridge.calculate(weights, activations, layer_name)
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores
