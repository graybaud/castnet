"""Sparsity measurement — migrated from legacy finetuning/sparsity.py."""

from typing import Dict, Any
import torch.nn as nn


def get_ffn_layers_generic(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Find all FFN linear layers in a model by name patterns.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of (name, nn.Linear) tuples.
    """
    ffn_patterns = ['fc1', 'fc2', 'c_fc', 'c_proj', 'gate_proj', 'up_proj', 'down_proj']
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p in name for p in ffn_patterns):
            layers.append((name, module))
    return layers


def get_sparsity_stats(model: nn.Module) -> Dict[str, Any]:
    """Measure sparsity of all FFN layers in a model.
    
    Returns:
        Dict with overall sparsity, per-layer breakdown, and total counts.
    """
    total_active, total_conn = 0, 0
    layer_sparsity = {}
    for name, module in get_ffn_layers_generic(model):
        active = (module.weight.data != 0).sum().item()
        total = module.weight.numel()
        total_active += active
        total_conn += total
        layer_sparsity[name] = (1 - active / total) * 100
    overall = (1 - total_active / total_conn) * 100 if total_conn > 0 else 0
    return {
        'overall': overall,
        'layers': layer_sparsity,
        'total_active': total_active,
        'total_conn': total_conn,
    }
