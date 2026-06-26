"""LoRA (Low-Rank Adaptation) for frozen-topology fine-tuning."""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA adapter for a linear layer.
    
    Freezes the original weights and adds trainable low-rank matrices.
    output = W @ x + (alpha/r) * A @ B @ x
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # Freeze original
        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A.float(), a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.float())
        self.lora_A.data = self.lora_A.data.to(dtype)
        self.lora_B.data = self.lora_B.data.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_linear(x)
        lora_out = self.dropout(x) @ self.lora_B.T @ self.lora_A.T * self.scaling
        return result + lora_out


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> list[str]:
    """Replace target linear layers with LoRA-augmented versions.
    
    Args:
        model: PyTorch model
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
        target_modules: List of module name substrings to target (default: FFN layers)
    
    Returns:
        List of replaced module names.
    """
    if target_modules is None:
        target_modules = ['fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']

    replaced = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            lora_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_module)
            replaced.append(name)

    print(f"Injected LoRA into {len(replaced)} modules")
    return replaced


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA weights back into original weights.
    
    After merging, the model is a standard nn.Module with no LoRA modules.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_weight = (module.lora_A @ module.lora_B) * module.scaling
            module.original_linear.weight.data += lora_weight.data
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.original_linear)

    print("LoRA weights merged")
