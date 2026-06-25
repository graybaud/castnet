"""APL-based scoring strategies â€” delegates to apl-pruning formulas.

All these strategies are available via APLFormulaStrategy with the formula name.

Usage:
    strategy = APLFormulaStrategy("gps_cube")
    scores = strategy.calculate(weights, activations, "layer1")

Available formulas (from apl-pruning):
    Built-in:
      magnitude, gradient, wanda, gps_local, direction, selectivity, distortion

    Extended (CastNet):
      gps_w_x, gps_w_grad, gps_cube,
      contrast_wanda, contrast_gradient,
      cos_w_x, cos_w_grad,
      union_wanda_grad, union_all3,
      sparsegps_score, gcs_score,
      chain_fc1, chain_fc2,
      softmax_grad_score, weighted_softmax_score,
      latency_score, symmetry_score, fractal_score,
      otsu_mask
"""

import torch
from torch import Tensor
from domain.scoring.strategies import ScoringStrategy
from infrastructure.scoring.apl_bridge import APLScoringBridge


class APLFormulaStrategy(ScoringStrategy):
    """Generic strategy driven by an APL formula string.

    Works with any formula registered in apl-pruning's METHODS dict.
    """

    def __init__(self, formula: str):
        self.bridge = APLScoringBridge(formula)

    def calculate(self, weights, activations, layer_name) -> Tensor:
        scores = self.bridge.calculate(weights, activations, layer_name)
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores


# Registry of all APL formulas available for CastNet
APL_FORMULAS = {
    # Core
    "magnitude": "|W|",
    "gradient": "|W| x |grad|",
    "wanda": "|W| x mean(|act|)",
    "gps_local": "gps_local",  # multi-line formula in apl-pruning

    # GPS variants
    "gps_w_x": "gps_w_x",
    "gps_w_grad": "gps_w_grad",
    "gps_cube": "gps_cube",

    # Contrastes
    "contrast_wanda": "contrast_wanda",
    "contrast_gradient": "contrast_gradient",

    # Alignements
    "cos_w_x": "cos_w_x",
    "cos_w_grad": "cos_w_grad",

    # Fusions
    "union_wanda_grad": "union_wanda_grad",
    "union_all3": "union_all3",

    # SparseGPS / GCS
    "sparsegps": "sparsegps_score",
    "gcs": "gcs_score",

    # Chain
    "chain_fc1": "chain_fc1",
    "chain_fc2": "chain_fc2",

    # Softmax variants
    "softmax_grad": "softmax_grad_score",
    "weighted_softmax": "weighted_softmax_score",

    # Q10-Q12
    "latency": "latency_score",
    "symmetry": "symmetry_score",
    "fractal": "fractal_score",

    # GPS components standalone
    "direction_only": "direction_standalone",
    "selectivity_only": "selectivity_standalone",
    "distortion_only": "distortion_standalone",

    # Wanda/Gradient x Distortion
    "wanda_x_distortion": "wanda_x_distortion",
    "gradient_x_distortion": "gradient_x_distortion",

    # Q30 — Hub scoring
    "q30_weighted": "q30_weighted",
    "q30_count": "q30_count",
}
