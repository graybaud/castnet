"""Logical rules R1-R6 — Pure domain logic.

These rules decide whether a neuron should be kept or pruned
based on geometric, gradient, correlation, distortion, and latency criteria.
"""

import torch
from torch import Tensor


def rule_r1_specialized(direction: Tensor, grad_norm: Tensor) -> Tensor:
    """R1: Specialized AND task-sensitive.
    
    Keeps neurons with high weight direction AND high gradient norm.
    """
    return (direction > 4.0) & (grad_norm > 0.001)


def rule_r3_unique(correlation: Tensor) -> Tensor:
    """R3: Unique (low correlation with peers).
    
    Keeps neurons that are not redundant with others.
    """
    return correlation < 0.4


def rule_r4_impact(distortion: Tensor) -> Tensor:
    """R4: High geometric impact.
    
    Keeps neurons whose removal would distort the semantic space.
    """
    return distortion > 0.001


def rule_r6_dynamic(latency: Tensor) -> Tensor:
    """R6: Strong temporal dynamics.
    
    Keeps neurons that activate in the middle of sequences.
    """
    return latency > 0.3


def apply_rules(
    direction: Tensor,
    grad_norm: Tensor,
    correlation: Tensor,
    distortion: Tensor,
    latency: Tensor,
) -> dict[str, Tensor]:
    """Apply all logical rules and return per-rule masks.
    
    Returns:
        dict with keys 'R1', 'R3', 'R4', 'R6' and their binary masks.
    """
    return {
        "R1": rule_r1_specialized(direction, grad_norm),
        "R3": rule_r3_unique(correlation),
        "R4": rule_r4_impact(distortion),
        "R6": rule_r6_dynamic(latency),
    }


def unify_rules(
    rule_masks: dict[str, Tensor],
    rule_weights: dict[str, float] | None = None,
) -> Tensor:
    """Combine rule masks into a single keep mask.
    
    A neuron is kept if it satisfies ANY rule (OR logic).
    
    Args:
        rule_masks: dict of rule_name -> binary mask
        rule_weights: optional per-rule weights for scoring
    
    Returns:
        Binary mask [d_out] where 1 = keep.
    """
    if rule_weights is None:
        rule_weights = {"R1": 1.0, "R3": 1.0, "R4": 1.0, "R6": 1.0}
    
    combined = torch.zeros_like(list(rule_masks.values())[0], dtype=torch.float32)
    for name, mask in rule_masks.items():
        combined += mask.float() * rule_weights.get(name, 1.0)
    
    return (combined > 0).float()


def rule_statistics(rule_masks: dict[str, Tensor], total_neurons: int) -> dict:
    """Count how many neurons are kept by each rule."""
    stats = {}
    for name, mask in rule_masks.items():
        stats[name] = {
            "kept": int(mask.sum().item()),
            "ratio": round(mask.sum().item() / total_neurons, 4),
        }
    return stats
