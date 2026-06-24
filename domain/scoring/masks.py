"""Binary mask generation — Pure domain logic."""

import torch
from torch import Tensor


def apply_percentile_mask(scores: Tensor, keep_fraction: float) -> Tensor:
    """Binary mask keeping the top keep_fraction of scores."""
    nonzero = scores[scores > 0].flatten()
    if len(nonzero) == 0:
        return torch.zeros_like(scores)

    if len(nonzero) > 10_000_000:
        idx = torch.randperm(len(nonzero))[:10_000_000]
        sample = nonzero[idx]
    else:
        sample = nonzero

    threshold = torch.quantile(sample.float(), 1.0 - keep_fraction)
    return (scores >= threshold).float()


def apply_masks_to_weights(
    masks: dict[str, Tensor],
    weight_provider,
) -> None:
    """Applies masks to weights (in-place multiplication)."""
    for name, mask in masks.items():
        W = weight_provider.get_weight(name)
        W.data *= mask.to(W.dtype)


def count_sparsity(masks: dict[str, Tensor]) -> dict:
    """Computes overall and per-layer sparsity."""
    total_active = 0
    total_conn = 0
    per_layer = {}

    for name, mask in masks.items():
        active = mask.sum().item()
        conn = mask.numel()
        total_active += active
        total_conn += conn
        per_layer[name] = round((1 - active / conn) * 100, 1)

    overall = round((1 - total_active / total_conn) * 100, 1) if total_conn > 0 else 0
    return {
        "overall_sparsity_pct": overall,
        "total_active": int(total_active),
        "total_connections": int(total_conn),
        "per_layer": per_layer,
    }
