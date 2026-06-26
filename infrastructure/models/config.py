"""Architecture detection & layer configuration — Infrastructure adapter.

Migrated from legacy extraction/config.py.
Uses transformers.AutoConfig for DeepSeek detection.
"""

import torch.nn as nn
from transformers import AutoConfig


def detect_architecture(model_or_name: str | object) -> str:
    """Detect model architecture family from name or model object.
    
    Returns: 'phi2', 'opt', 'llama', 'deepseek_v2', 'deepseek_moe', 'generic'
    """
    if isinstance(model_or_name, str):
        name = model_or_name.lower()
    else:
        name = str(type(model_or_name)).lower()

    if 'deepseek' in name:
        if 'v2' in name or 'v3' in name:
            return 'deepseek_v2'
        if 'moe' in name or 'expert' in name:
            return 'deepseek_moe'
        return 'deepseek_moe'
    if 'phi-2' in name or 'phi2' in name:
        return 'phi2'
    if 'opt' in name:
        return 'opt'
    if 'llama' in name:
        return 'llama'
    return 'generic'


def is_deepseek_v2(arch: str) -> bool:
    """Check if architecture is DeepSeek V2 or V3."""
    return arch in ('deepseek_v2', 'deepseek_v3')


def get_deepseek_v2_config(model_name: str) -> dict:
    """Get DeepSeek V2/V3 model configuration from HuggingFace config."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return {
        "hidden_size": getattr(config, 'hidden_size', 2048),
        "moe_intermediate_size": getattr(config, 'moe_intermediate_size', 8192),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', 27),
        "num_experts": getattr(config, 'n_routed_experts', 64),
        "n_shared_experts": getattr(config, 'n_shared_experts', 1),
    }


def get_ffn_pairs(model, architecture: str | None = None) -> list:
    """Get FFN layer pairs (fc1/fc2 or gate_proj/down_proj).
    
    Args:
        model: HuggingFace model
        architecture: detected architecture string
    
    Returns:
        List of (fc1_name, fc1_module, fc2_name, fc2_module) tuples.
    """
    if architecture is None:
        architecture = detect_architecture(model)

    if architecture == 'phi2':
        return [
            (f"layer{i}.fc1", layer.mlp.fc1, f"layer{i}.fc2", layer.mlp.fc2)
            for i, layer in enumerate(model.model.layers)
        ]

    if architecture == 'opt':
        return [
            (f"layer{i}.fc1", layer.fc1, f"layer{i}.fc2", layer.fc2)
            for i, layer in enumerate(model.model.decoder.layers)
        ]

    if architecture == 'llama':
        pairs = []
        for i, layer in enumerate(model.model.layers):
            pairs.append((
                f"layer{i}.gate_proj", layer.mlp.gate_proj,
                f"layer{i}.down_proj", layer.mlp.down_proj,
            ))
            pairs.append((
                f"layer{i}.up_proj", layer.mlp.up_proj,
                f"layer{i}.down_proj", layer.mlp.down_proj,
            ))
        return pairs

    if architecture in ('deepseek_moe', 'deepseek_v2', 'deepseek_v3'):
        pairs = []
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'experts'):
                for e_idx, expert in enumerate(layer.mlp.experts):
                    if hasattr(expert, 'gate_proj') and hasattr(expert, 'down_proj'):
                        pairs.append((
                            f"layer{i}.expert{e_idx}.gate_proj", expert.gate_proj,
                            f"layer{i}.expert{e_idx}.down_proj", expert.down_proj,
                        ))
                    if hasattr(expert, 'up_proj') and hasattr(expert, 'down_proj'):
                        pairs.append((
                            f"layer{i}.expert{e_idx}.up_proj", expert.up_proj,
                            f"layer{i}.expert{e_idx}.down_proj", expert.down_proj,
                        ))
        return pairs

    # Generic fallback
    all_mods = dict(model.named_modules())
    pairs = []
    for name, m in all_mods.items():
        if isinstance(m, nn.Linear) and ('fc1' in name or 'c_fc' in name):
            fc2 = name.replace('fc1', 'fc2').replace('c_fc', 'c_proj')
            if fc2 in all_mods:
                pairs.append((
                    name.replace('.', '_'), m,
                    fc2.replace('.', '_'), all_mods[fc2],
                ))
    return pairs


def get_attention_layers(model, architecture: str | None = None) -> list:
    """Get attention projection layers (Q, K, V, O).
    
    Returns:
        List of (name, module) tuples.
    """
    if architecture is None:
        architecture = detect_architecture(model)

    layers = []

    if architecture == 'phi2':
        for i, layer in enumerate(model.model.layers):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                if hasattr(layer.self_attn, proj):
                    layers.append((f"layer{i}.attn.{proj}", getattr(layer.self_attn, proj)))

    elif architecture == 'opt':
        for i, layer in enumerate(model.model.decoder.layers):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                if hasattr(layer.self_attn, proj):
                    layers.append((f"layer{i}.attn.{proj}", getattr(layer.self_attn, proj)))

    elif architecture == 'llama':
        for i, layer in enumerate(model.model.layers):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, proj):
                    layers.append((f"layer{i}.attn.{proj}", getattr(layer.self_attn, proj)))

    return layers
