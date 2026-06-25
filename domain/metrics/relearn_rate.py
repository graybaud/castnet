"""Relearning rate metric — Pure domain logic."""

import torch
from torch import Tensor


def compute_weight_change(
    weights_before: dict[str, Tensor],
    weights_after: dict[str, Tensor],
) -> dict:
    """Compute how much weights changed after partial reset and retraining.

    Args:
        weights_before: Weights before reset.
        weights_after: Weights after retraining.

    Returns:
        dict with average relative change and per-layer breakdown.
    """
    total_change = 0.0
    total_params = 0
    per_layer = {}

    for name in weights_before:
        if name not in weights_after:
            continue
        W_before = weights_before[name]
        W_after = weights_after[name]

        mask = (W_before != 0)
        if mask.sum() == 0:
            per_layer[name] = 0.0
            continue

        delta = (W_after - W_before).abs()
        rel_change = (delta / (W_before.abs() + 1e-8))[mask].mean().item()

        per_layer[name] = round(rel_change, 6)
        total_change += delta.sum().item()
        total_params += mask.sum().item()

    avg_rel_change = total_change / max(total_params, 1)
    return {
        "avg_relative_weight_change": round(avg_rel_change, 6),
        "total_params": total_params,
        "per_layer": per_layer,
    }
