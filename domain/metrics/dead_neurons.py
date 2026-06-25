"""Dead neuron analysis — Pure domain logic."""

import torch
from torch import Tensor


def count_dead_neurons(weight: Tensor) -> dict:
    """Count neurons with all-zero incoming connections.

    Args:
        weight: Weight matrix [d_out, d_in]

    Returns:
        dict with dead neuron count and ratio.
    """
    d_out, d_in = weight.shape
    dead = (weight.abs().sum(dim=1) == 0).sum().item()
    return {
        "total_neurons": d_out,
        "dead_neurons": dead,
        "dead_ratio": dead / d_out if d_out > 0 else 0.0,
        "active_neurons": d_out - dead,
        "active_ratio": (d_out - dead) / d_out if d_out > 0 else 0.0,
    }


def count_dead_neurons_all_layers(weights: dict[str, Tensor]) -> dict:
    """Count dead neurons across all layers."""
    total = 0
    total_dead = 0
    per_layer = {}
    for name, W in weights.items():
        stats = count_dead_neurons(W)
        per_layer[name] = stats
        total += stats["total_neurons"]
        total_dead += stats["dead_neurons"]
    return {
        "total_neurons": total,
        "total_dead": total_dead,
        "dead_ratio_global": total_dead / total if total > 0 else 0.0,
        "per_layer": per_layer,
    }
