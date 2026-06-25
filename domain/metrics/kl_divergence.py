"""KL Divergence between dense and sparse model logits — Pure domain logic."""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_kl_divergence(logits_sparse: Tensor, logits_dense: Tensor) -> float:
    """Compute KL divergence between two sets of logits.

    KL(P_dense || P_sparse) = sum(P_dense * log(P_dense / P_sparse))

    Args:
        logits_sparse: Logits from sparse model [N, vocab_size]
        logits_dense: Logits from dense model [N, vocab_size]

    Returns:
        KL divergence (scalar).
    """
    log_probs_sparse = F.log_softmax(logits_sparse, dim=-1)
    probs_dense = F.softmax(logits_dense, dim=-1)
    return F.kl_div(log_probs_sparse, probs_dense, reduction="batchmean").item()


def compute_kl_divergence_batched(
    logits_sparse_list: list[Tensor],
    logits_dense_list: list[Tensor],
) -> dict:
    """Compute KL divergence statistics over multiple batches.

    Returns:
        dict with mean, std, and per-batch values.
    """
    kl_values = []
    for logits_s, logits_d in zip(logits_sparse_list, logits_dense_list):
        kl_values.append(compute_kl_divergence(logits_s, logits_d))

    if not kl_values:
        return {"mean": None, "std": None, "n_samples": 0}

    t = torch.tensor(kl_values)
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "n_samples": len(kl_values),
    }
