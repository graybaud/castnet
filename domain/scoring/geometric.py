"""Geometric utilities — Pure domain logic for GPS scoring."""

import torch
from torch import Tensor


def cosine_similarity_matrix(vectors: Tensor) -> Tensor:
    """Compute NxN cosine similarity matrix from N vectors.
    
    Args:
        vectors: [N, D] tensor of N vectors of dimension D
    
    Returns:
        [N, N] similarity matrix.
    """
    v = vectors.float()
    v_norm = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    return v_norm @ v_norm.T


def frobenius_distortion(S1: Tensor, S2: Tensor) -> Tensor:
    """Normalized Frobenius distance between two similarity matrices.
    
    distortion = ||S1 - S2||_F / ||S1||_F
    """
    return (S1 - S2).norm() / (S1.norm() + 1e-8)


def _organize_layers(ffn_layers: list[tuple[str, object]]) -> dict:
    """Organize layer names by layer index.
    
    Args:
        ffn_layers: list of (name, module) tuples
    
    Returns:
        dict of layer_index -> {'fc1': name, 'fc2': name}
    """
    layers = {}
    for name, _ in ffn_layers:
        parts = name.replace('.', '_').split('_')
        for p in parts:
            if p.startswith('layer') and p.replace('layer', '').isdigit():
                layer_idx = int(p.replace('layer', ''))
                if layer_idx not in layers:
                    layers[layer_idx] = {}
                if 'fc1' in name or 'gate_proj' in name or 'up_proj' in name:
                    layers[layer_idx]['fc1'] = name
                elif 'fc2' in name or 'down_proj' in name:
                    layers[layer_idx]['fc2'] = name
                break
    return layers


def find_natural_threshold(
    scores: dict[str, Tensor],
    num_bins: int = 100,
) -> dict:
    """Find natural sparsity threshold using Otsu's method.
    
    Args:
        scores: dict of layer_name -> score_tensor
        num_bins: number of histogram bins
    
    Returns:
        dict with threshold, keep_fraction, method, is_bimodal.
    """
    all_scores = torch.cat([s.flatten() for s in scores.values()])
    all_scores = all_scores[all_scores > 1e-8]
    if len(all_scores) < 100:
        return {"threshold": 0.5, "keep_fraction": 0.5, "method": "fallback", "is_bimodal": False}

    hist = torch.histc(all_scores, bins=num_bins, min=0, max=1)
    bin_centers = torch.linspace(0, 1, num_bins)
    total = hist.sum()

    best_threshold = 0.5
    best_variance = 0.0
    for i in range(1, num_bins):
        w1 = hist[:i].sum() / total
        w2 = hist[i:].sum() / total
        if w1 == 0 or w2 == 0:
            continue
        m1 = (hist[:i] * bin_centers[:i]).sum() / hist[:i].sum()
        m2 = (hist[i:] * bin_centers[i:]).sum() / hist[i:].sum()
        var_between = w1 * w2 * (m1 - m2) ** 2
        if var_between > best_variance:
            best_variance = var_between
            best_threshold = bin_centers[i].item()

    threshold = best_threshold
    above = (all_scores >= threshold).sum().item()
    keep_fraction = above / len(all_scores) if len(all_scores) > 0 else 0.5

    hist_np = hist.numpy()
    peaks = sum(
        1 for i in range(1, len(hist_np) - 1)
        if hist_np[i] > hist_np[i - 1]
        and hist_np[i] > hist_np[i + 1]
        and hist_np[i] > hist_np.mean()
    )

    return {
        "threshold": round(threshold, 4),
        "keep_fraction": round(keep_fraction, 4),
        "method": "otsu_fast",
        "total_connections_scored": len(all_scores),
        "is_bimodal": peaks >= 2,
        "num_peaks": peaks,
    }


def neuron_direction(weight_vector: torch.Tensor) -> float:
    """Ratio max/mean of absolute weights."""
    w_abs = weight_vector.float().abs()
    w_mean = w_abs.mean()
    return (w_abs.max() / w_mean).item() if w_mean > 0 else 0.0


def neuron_selectivity(activations: torch.Tensor) -> float:
    """Variance/mean of activations."""
    a = activations.float()
    mean = a.mean()
    return (a.var() / mean).item() if mean > 0 else 0.0
