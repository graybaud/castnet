"""APL-based scoring strategies."""

import torch
from torch import Tensor
from domain.scoring.strategies import ScoringStrategy
from infrastructure.scoring.apl_bridge import APLScoringBridge


class APLFormulaStrategy(ScoringStrategy):
    """Generic strategy driven by an APL formula string."""

    def __init__(self, formula: str):
        self.bridge = APLScoringBridge(formula)
        self.needs_grad = self._detect_needs_grad(formula)

    def _detect_needs_grad(self, formula: str) -> bool:
        grad_keywords = ["gradient", "grad", "chain_fc1", "chain_fc2",
                        "softmax_grad", "gps_w_grad", "gps_cube",
                        "contrast_gradient", "cos_w_grad", "union_all3",
                        "union_wanda_grad", "gradient_x_distortion"]
        return any(kw in formula for kw in grad_keywords)

    def calculate(self, weights, activations, layer_name) -> Tensor:
        scores = self.bridge.calculate(weights, activations, layer_name)
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


APL_FORMULAS = {
    "magnitude": "|W|",
    "gradient": "|W| x |grad|",
    "wanda": "|W| x mean(|act|)",
    "gps_local": "gps_local",
    "gps_w_x": "gps_w_x",
    "gps_w_grad": "gps_w_grad",
    "gps_cube": "gps_cube",
    "contrast_wanda": "contrast_wanda",
    "contrast_gradient": "contrast_gradient",
    "cos_w_x": "cos_w_x",
    "cos_w_grad": "cos_w_grad",
    "union_wanda_grad": "union_wanda_grad",
    "union_all3": "union_all3",
    "sparsegps": "sparsegps_score",
    "gcs": "gcs_score",
    "chain_fc1": "chain_fc1",
    "chain_fc2": "chain_fc2",
    "softmax_grad": "softmax_grad_score",
    "weighted_softmax": "weighted_softmax_score",
    "latency": "latency_score",
    "symmetry": "symmetry_score",
    "fractal": "fractal_score",
    "direction_only": "direction_per_neuron",
    "selectivity_only": "selectivity",
    "distortion_only": "distortion_standalone",
    "wanda_x_distortion": "wanda_x_distortion",
    "gradient_x_distortion": "gradient_x_distortion",
    "q30_weighted": "q30_weighted",
    "q30_count": "q30_count",
}
