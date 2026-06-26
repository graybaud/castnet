"""DeepSeek-V2/V3 Low-RAM Expert Loader — Infrastructure adapter."""

import os
import re
import json
from collections import defaultdict
import torch
import torch.nn as nn
from safetensors import safe_open


def load_deepseek_v2_expert_weights(
    model_name: str,
    expert_indices: list[int] | None = None,
    device: str = 'cpu',
) -> dict | None:
    """Load DeepSeek-V2 expert weights from HF cache.

    Args:
        model_name: HuggingFace model name
        expert_indices: List of expert indices to load (None = all)
        device: Torch device

    Returns:
        Nested dict: layer_idx -> expert_idx -> projection_name -> weight_tensor
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(
        cache_dir, f"models--{model_name.replace('/', '--')}"
    )
    if not os.path.exists(model_dir):
        print("[!] Model not cached")
        return None

    snapshots_dir = os.path.join(model_dir, "snapshots")
    snapshots = os.listdir(snapshots_dir)
    if not snapshots:
        return None
    snapshot_path = os.path.join(snapshots_dir, snapshots[0])

    index_file = next(
        (os.path.join(snapshot_path, f) for f in os.listdir(snapshot_path)
         if f.endswith('.index.json')),
        None,
    )
    if not index_file:
        return None

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get('weight_map', {})
    expert_files = defaultdict(lambda: defaultdict(list))
    pattern = re.compile(
        r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight'
    )

    for wn, fn in weight_map.items():
        m = pattern.search(wn)
        if m:
            layer_idx = int(m.group(1))
            expert_idx = int(m.group(2))
            proj_name = m.group(3)
            if (expert_indices is None or expert_idx in expert_indices) and \
               proj_name in ('gate_proj', 'up_proj', 'down_proj'):
                expert_files[fn][(layer_idx, expert_idx, proj_name)] = wn

    results = defaultdict(lambda: defaultdict(dict))
    for fn, wdict in expert_files.items():
        fp = os.path.join(snapshot_path, fn)
        if os.path.exists(fp):
            with safe_open(fp, framework="pt") as f:
                for (li, ei, pn), wn in wdict.items():
                    if wn in f.keys():
                        results[li][ei][pn] = f.get_tensor(wn).float().to(device)

    return dict(results)


class DeepSeekV2ExpertWrapper(nn.Module):
    """Wraps a single DeepSeek-V2 expert as a standalone module."""

    def __init__(self, expert_weights: dict, hidden_size: int, moe_int: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, moe_int, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_int, bias=False)
        self.down_proj = nn.Linear(moe_int, hidden_size, bias=False)

        for p in ['gate_proj', 'up_proj', 'down_proj']:
            if p in expert_weights:
                getattr(self, p).weight.data = expert_weights[p]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def get_weight(self, proj: str) -> torch.Tensor:
        return getattr(self, proj).weight
