"""Scale-free topology exponent (gamma) — Pure domain logic."""

import numpy as np
import torch.nn as nn


def measure_gamma(model: nn.Module, threshold: float = 0.0) -> float:
    """Measure the power-law exponent (gamma) of the degree distribution
    across all FFN weight matrices in the model.

    Gamma > 2.0 indicates a scale-free topology (robust to random failures).
    """
    from infrastructure.models.huggingface import HuggingFaceWeightProvider
    FFN_PATTERNS = HuggingFaceWeightProvider.FFN_PATTERNS

    degrees = []
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if any(p in name_lower for p in FFN_PATTERNS) and param.dim() == 2:
            W = param.detach().float()
            mask = W.abs() > threshold
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
