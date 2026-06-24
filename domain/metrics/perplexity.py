"""Perplexity metrics — Pure domain logic."""

import math
import torch
from torch import Tensor


def compute_perplexity_from_loss(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.
    
    perplexity = exp(loss)
    """
    return math.exp(loss) if loss < 10 else float("inf")


def compute_perplexity_from_logits(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity from logits and target tokens.
    
    Args:
        logits: [B, S, V] or [N, V]
        targets: [B, S] or [N]
        ignore_index: padding token id to ignore
    
    Returns:
        Perplexity (float).
    """
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    return compute_perplexity_from_loss(loss.item())


def compute_perplexity_from_total(total_loss: float, total_tokens: int) -> float:
    """Compute perplexity from accumulated loss and token count."""
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return compute_perplexity_from_loss(avg_loss)
