"""Weight distribution analysis — Pure domain logic."""

import torch
from torch import Tensor


def analyze_weight_distribution(weight: Tensor, num_bins: int = 20) -> dict:
    """Analyze the distribution of non-zero weights.

    Args:
        weight: Weight matrix [d_out, d_in]
        num_bins: Number of histogram bins

    Returns:
        dict with statistics about weight distribution.
    """
    nonzero = weight[weight != 0].float()
    zero_count = (weight == 0).sum().item()
    total = weight.numel()

    if len(nonzero) == 0:
        return {
            "total_weights": total,
            "nonzero_weights": 0,
            "sparsity": 1.0,
        }

    return {
        "total_weights": total,
        "nonzero_weights": len(nonzero),
        "zero_weights": zero_count,
        "sparsity": zero_count / total,
        "mean_abs": nonzero.abs().mean().item(),
        "std_abs": nonzero.abs().std().item(),
        "min_abs": nonzero.abs().min().item(),
        "max_abs": nonzero.abs().max().item(),
        "median_abs": nonzero.abs().median().item(),
    }
