"""Score I/O utilities — Load/save scores, infer method from filename."""

import os
import torch


def infer_method_from_filename(filepath: str) -> str:
    """Infer extraction method from score filename.
    
    Examples:
        "phi2_wanda_scores_b100.pt" → "wanda"
        "opt_gradient_scores.pt" → "gradient"
    """
    basename = os.path.basename(filepath).lower()
    for m in ['wanda_chain', 'wanda', 'chain', 'gradient', 'gps']:
        if m in basename:
            return m
    return 'unknown'


def load_scores(scores_path: str, device: str = "cpu") -> dict:
    """Load pre-computed scores from .pt or .safetensors file.
    
    Args:
        scores_path: Path to scores file
        device: Torch device
    
    Returns:
        dict of layer_name -> score_tensor
    """
    if scores_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(scores_path)

    # .pt format
    try:
        scores = torch.load(scores_path, map_location=device, weights_only=True)
    except Exception:
        scores = torch.load(scores_path, map_location=device, weights_only=False)

    return scores
