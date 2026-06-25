"""Scale-free topology exponent (gamma) — Pure domain logic."""

import numpy as np
from domain.constants import is_ffn_layer


def measure_gamma_from_weights(weights: dict) -> float:
    """Measure the power-law exponent (gamma) of the degree distribution.

    Args:
        weights: dict of layer_name -> weight_tensor [d_out, d_in]

    Returns:
        Gamma exponent (float), or 0.0 if insufficient data.
    """
    degrees = []
    for name, W in weights.items():
        if not is_ffn_layer(name):
            continue
        if W.dim() != 2:
            continue
        mask = W.abs() > 0.0
        col_degrees = mask.sum(dim=0).tolist()
        degrees.extend(col_degrees)

    if len(degrees) < 10:
        return 0.0

    degrees = np.array([d for d in degrees if d > 0])
    if len(degrees) < 10:
        return 0.0

    try:
        hist, edges = np.histogram(degrees, bins=50)
        centers = (edges[:-1] + edges[1:]) / 2
        positive_mask = hist > 0
        if positive_mask.sum() < 5:
            return 0.0
        coeffs = np.polyfit(
            np.log10(centers[positive_mask]),
            np.log10(hist[positive_mask]),
            1,
        )
        return round(-coeffs[0], 3)
    except Exception:
        return 0.0
