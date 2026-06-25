"""Rapid adaptation metric — Pure domain logic."""

import torch
from torch import Tensor


def compute_adaptation_rate(
    loss_curve: list[float],
    window_size: int = 20,
) -> dict:
    """Compute how fast a model adapts from its loss curve.

    Args:
        loss_curve: List of loss values over training steps.
        window_size: Number of recent steps to fit.

    Returns:
        dict with initial_loss, final_loss, loss_ratio, decrease_rate.
    """
    if len(loss_curve) < window_size:
        return {
            "initial_loss": loss_curve[0] if loss_curve else None,
            "final_loss": loss_curve[-1] if loss_curve else None,
            "loss_ratio": 1.0,
            "decrease_rate": 0.0,
        }

    initial = sum(loss_curve[:5]) / 5
    final = sum(loss_curve[-window_size:]) / window_size

    recent = loss_curve[-window_size:]
    import numpy as np
    steps = np.arange(len(recent))
    slope, _ = np.polyfit(steps, recent, 1)

    return {
        "initial_loss": round(initial, 4),
        "final_loss": round(final, 4),
        "loss_ratio": round(final / max(initial, 1e-8), 4),
        "decrease_rate_per_step": round(-slope.item() * 100, 6),
        "total_steps": len(loss_curve),
    }
