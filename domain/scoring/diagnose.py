"""Score diagnostics — Pure domain logic. No framework imports."""

import torch


def diagnose_scores(scores: dict, max_layers: int = 5) -> dict:
    """Analyze per-layer score distribution.
    
    Args:
        scores: dict of layer_name -> score_tensor
        max_layers: max non-zero layers to show in output
    
    Returns:
        dict with zero_layers, non_zero_layers, summary.
    """
    if not scores:
        return {"zero_layers": 0, "total_layers": 0, "all_zero": True}

    zero_layers = []
    non_zero = []

    for name, score in scores.items():
        s = score.flatten()
        nz = s[s > 0]
        if len(nz) > 0:
            non_zero.append({
                "name": name,
                "total": len(s),
                "non_zero": len(nz),
                "non_zero_pct": round(len(nz) / len(s) * 100, 1),
                "q50": round(torch.median(nz).item(), 6),
            })
        else:
            zero_layers.append(name)

    return {
        "total_layers": len(scores),
        "zero_layers": len(zero_layers),
        "zero_layer_names": zero_layers,
        "non_zero_layers": non_zero[:max_layers],
        "all_zero": len(zero_layers) == len(scores),
    }
