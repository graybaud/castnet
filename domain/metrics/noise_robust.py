"""Noise robustness metric — Pure domain logic."""

import torch
from torch import Tensor


def compute_noise_degradation(
    perplexity_clean: float,
    perplexities_noisy: dict[str, float],
) -> dict:
    """Compute perplexity degradation under different noise levels.

    Args:
        perplexity_clean: Perplexity without noise.
        perplexities_noisy: Dict of noise_level -> perplexity.

    Returns:
        dict with degradation ratios per noise level.
    """
    results = {}
    for level, perp_noisy in perplexities_noisy.items():
        degradation = (perp_noisy - perplexity_clean) / max(perplexity_clean, 1.0)
        results[str(level)] = {
            "perplexity": round(perp_noisy, 2),
            "degradation_ratio": round(degradation, 4),
            "perplexity_increase": round(perp_noisy - perplexity_clean, 2),
        }

    return {
        "clean_perplexity": round(perplexity_clean, 2),
        "noise_levels": results,
    }
