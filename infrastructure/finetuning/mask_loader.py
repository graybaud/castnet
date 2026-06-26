"""Mask loading and application — migrated from legacy finetuning/masks.py."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from safetensors.torch import load_file


def load_masks(mask_source: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load masks from .safetensors or .pt checkpoint.
    
    Args:
        mask_source: Path to .safetensors or .pt file
        device: Torch device
    
    Returns:
        Dict of mask_name -> mask_tensor
    """
    if mask_source.endswith('.safetensors'):
        masks = load_file(mask_source)
        return {name: m.float().to(device) for name, m in masks.items()}
    elif mask_source.endswith('.pt'):
        try:
            ckpt = torch.load(mask_source, map_location='cpu', weights_only=True)
        except Exception:
            ckpt = torch.load(mask_source, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        masks = {}
        for name, tensor in state_dict.items():
            if tensor.dim() >= 2 and 'weight' in name:
                mask = (tensor != 0).float()
                if mask.sum() < mask.numel():
                    mask_name = name.replace('.weight', '').replace('.', '_')
                    masks[mask_name] = mask.to(device)
        return masks
    else:
        raise ValueError(f"Unsupported mask file format: {mask_source}")


def convert_mask_name_to_model(name: str, architecture: str) -> list[str]:
    """Convert a mask name (with underscores) to potential model module paths.
    
    Args:
        name: Mask name like 'layer0_fc1'
        architecture: Model architecture ('opt', 'phi2', 'llama', 'generic')
    
    Returns:
        List of candidate module paths in the model.
    """
    parts = name.replace('.', '_').split('_')
    layer_num = None
    fc_type = None

    for p in parts:
        if p.startswith('layer') and p.replace('layer', '').isdigit():
            layer_num = p.replace('layer', '')
        if p in ['fc1', 'fc2', 'c_fc', 'c_proj', 'gate_proj', 'up_proj', 'down_proj']:
            fc_type = p

    # Check 2-token combinations (gate_proj -> gate + proj)
    if fc_type is None:
        for i in range(len(parts) - 1):
            combined = parts[i] + "_" + parts[i + 1]
            if combined in ["gate_proj", "up_proj", "down_proj", "c_fc", "c_proj"]:
                fc_type = combined
                break

    if layer_num is None or fc_type is None:
        return [name]

    if architecture == 'opt':
        return [f"model.decoder.layers.{layer_num}.{fc_type}"]
    elif architecture == 'phi2':
        return [f"model.layers.{layer_num}.mlp.{fc_type}"]
    elif architecture == 'llama':
        return [f"model.layers.{layer_num}.mlp.{fc_type}"]
    else:
        return [
            f"model.decoder.layers.{layer_num}.{fc_type}",
            f"model.layers.{layer_num}.mlp.{fc_type}",
            f"model.layers.{layer_num}.{fc_type}",
        ]


def build_mask_application_dict(
    model: nn.Module, masks: Dict[str, torch.Tensor], architecture: str
) -> Dict[str, Tuple[nn.Linear, torch.Tensor]]:
    """Map mask names to model modules.
    
    Returns:
        Dict of mask_name -> (nn.Linear module, mask_tensor)
    """
    model_modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    applied = {}
    for mask_name, mask_tensor in masks.items():
        candidates = convert_mask_name_to_model(mask_name, architecture)
        found = False
        for candidate in candidates:
            if candidate in model_modules:
                applied[mask_name] = (model_modules[candidate], mask_tensor)
                found = True
                break
        if found:
            continue
        if mask_name in model_modules:
            applied[mask_name] = (model_modules[mask_name], mask_tensor)
            continue
        for model_name, module in model_modules.items():
            if model_name.replace('.', '_') == mask_name:
                applied[mask_name] = (module, mask_tensor)
                break
        else:
            print(f"  WARNING: Module not found for mask '{mask_name}'. Skipping.")
    return applied


def apply_masks(model: nn.Module, mask_dict: Dict[str, Tuple[nn.Linear, torch.Tensor]]) -> None:
    """Apply masks to model weights in-place."""
    for _, (module, mask) in mask_dict.items():
        module.weight.data *= mask.to(module.weight.dtype)
