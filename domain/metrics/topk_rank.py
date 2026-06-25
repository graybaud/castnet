"""Top-K rank correlation between dense and sparse model predictions."""

import torch
from torch import Tensor


def topk_overlap(logits_sparse: Tensor, logits_dense: Tensor, k: int = 10) -> float:
    """Compute overlap between top-k predictions of two models.

    Args:
        logits_sparse: Logits from sparse model [vocab_size]
        logits_dense: Logits from dense model [vocab_size]
        k: Number of top predictions to compare

    Returns:
        Overlap ratio (0.0 to 1.0).
    """
    top_sparse = set(torch.topk(logits_sparse, k).indices.tolist())
    top_dense = set(torch.topk(logits_dense, k).indices.tolist())
    return len(top_sparse & top_dense) / k


def topk_overlap_batched(
    logits_sparse_list: list[Tensor],
    logits_dense_list: list[Tensor],
    k_values: list[int] = [5, 10],
) -> dict:
    """Compute top-k overlap statistics over multiple batches.

    Returns:
        dict with per-k mean and std.
    """
    results = {k: [] for k in k_values}
    for logits_s, logits_d in zip(logits_sparse_list, logits_dense_list):
        # Take the last token of each sequence
        s = logits_s[0, -1, :]
        d = logits_d[0, -1, :]
        for k in k_values:
            results[k].append(topk_overlap(s, d, k))

    output = {}
    for k in k_values:
        if results[k]:
            t = torch.tensor(results[k])
            output[f"top{k}_overlap_mean"] = t.mean().item()
            output[f"top{k}_overlap_std"] = t.std().item()
    output["n_samples"] = len(results[k_values[0]]) if results[k_values[0]] else 0
    return output
