"""Perplexity metrics — Pure domain logic."""

import math


def compute_perplexity_from_loss(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.
    
    perplexity = exp(loss), clamped to inf if loss > 10.
    """
    return math.exp(loss) if loss < 10 else float("inf")


def compute_perplexity_from_total(total_loss: float, total_tokens: int) -> float:
    """Compute perplexity from accumulated loss and token count.
    
    perplexity = exp(total_loss / total_tokens)
    """
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return compute_perplexity_from_loss(avg_loss)
