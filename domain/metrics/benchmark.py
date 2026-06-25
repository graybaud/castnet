"""Magnitude pruning benchmark — Pure domain logic."""

import torch
from torch import Tensor


def apply_magnitude_pruning(weight: Tensor, threshold: float) -> tuple[Tensor, float]:
    """Apply magnitude pruning to a single weight matrix.

    Zeros out all weights with absolute value below threshold.

    Args:
        weight: Weight matrix [d_out, d_in]
        threshold: Pruning threshold

    Returns:
        (pruned_weight, sparsity) tuple.
    """
    mask = weight.abs() > threshold
    pruned = weight * mask.float()
    sparsity = 1.0 - mask.sum().item() / mask.numel()
    return pruned, sparsity


def apply_magnitude_pruning_all_layers(
    weights: dict[str, Tensor],
    threshold: float,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Apply magnitude pruning to all layers.

    Returns:
        (pruned_weights, per_layer_sparsity) tuple.
    """
    pruned = {}
    sparsities = {}
    total_active = 0
    total_conn = 0

    for name, W in weights.items():
        pruned[name], sp = apply_magnitude_pruning(W, threshold)
        sparsities[name] = sp
        total_active += (W.abs() > threshold).sum().item()
        total_conn += W.numel()

    overall_sparsity = 1.0 - total_active / total_conn if total_conn > 0 else 0.0
    sparsities["overall"] = overall_sparsity

    return pruned, sparsities


def compute_magnitude_benchmark(
    weights: dict[str, Tensor],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Benchmark magnitude pruning at multiple thresholds.

    Args:
        weights: Dict of layer_name -> weight matrix
        thresholds: List of thresholds to test (default: 0.01 to 0.30)

    Returns:
        List of dicts with threshold, sparsity_pct.
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.30]

    results = []
    for t in thresholds:
        _, sparsities = apply_magnitude_pruning_all_layers(weights, t)
        results.append({
            "threshold": t,
            "sparsity_pct": round(sparsities["overall"] * 100, 2),
        })

    return results
