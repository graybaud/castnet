"""Neuron activation frequency analysis — Pure domain logic."""

import torch
from torch import Tensor


def compute_activation_frequency(
    activations: Tensor,
    threshold: float = 0.0,
) -> dict:
    """Compute how often each neuron fires.

    Args:
        activations: Activation tensor [N_tokens, d_out]
        threshold: Value above which a neuron is considered "active"

    Returns:
        dict with per-neuron frequencies and dead neuron stats.
    """
    if activations.dim() == 3:
        activations = activations.reshape(-1, activations.shape[-1])

    n_tokens, d_out = activations.shape
    active = (activations > threshold).float()
    freqs = active.sum(dim=0) / n_tokens  # [d_out]

    dead = (freqs == 0).sum().item()

    alive = freqs[freqs > 0]

    return {
        "neurons": d_out,
        "dead_neurons": dead,
        "dead_ratio": dead / d_out,
        "mean_frequency": alive.mean().item() if len(alive) > 0 else 0.0,
        "median_frequency": alive.median().item() if len(alive) > 0 else 0.0,
        "max_frequency": freqs.max().item(),
        "tokens_processed": n_tokens,
    }


def compute_activation_frequency_all_layers(
    activations_dict: dict[str, Tensor],
    threshold: float = 0.0,
) -> dict:
    """Compute activation frequency for all layers.

    Args:
        activations_dict: Dict of layer_name -> activations [N, d_out]
        threshold: Activation threshold

    Returns:
        dict with global stats and per-layer breakdown.
    """
    total_neurons = 0
    total_dead = 0
    per_layer = {}

    for name, acts in activations_dict.items():
        stats = compute_activation_frequency(acts, threshold)
        per_layer[name] = stats
        total_neurons += stats["neurons"]
        total_dead += stats["dead_neurons"]

    return {
        "total_neurons": total_neurons,
        "total_dead_neurons": total_dead,
        "dead_ratio_global": total_dead / max(total_neurons, 1),
        "per_layer": per_layer,
    }
