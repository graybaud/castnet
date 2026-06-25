"""HuggingFace adapter — Implements WeightProvider."""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from domain.scoring.ports import WeightProvider

# Disable HF Hub requests for faster loading
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class HuggingFaceWeightProvider(WeightProvider):
    """Provides weights and gradients from a HuggingFace model."""

    from domain.constants import FFN_PATTERNS

    def __init__(self, model_name: str, device: str = "cuda"):
        # Force local files only — no network requests
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if "cuda" in device else None,
            local_files_only=True,  # ← NO NETWORK
        )
        self._device = device
        self.model.train()  # Set training mode once
        self._ffn_layers = self._detect_ffn_layers()
        self._name_map = {n.replace(".", "_"): n for n in self._ffn_layers}

    @property
    def _model(self):
        return self.model

    def _detect_ffn_layers(self) -> dict[str, nn.Module]:
        layers = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                if not hasattr(module, 'weight') or not hasattr(module, 'bias'):
                    continue
                if module.weight.dim() != 2:
                    continue
            name_lower = name.lower()
            if "attn" in name_lower or "attention" in name_lower or "lm_head" in name_lower:
                continue
            if any(p in name_lower for p in self.FFN_PATTERNS):
                layers[name] = module
        return layers

    def _resolve_name(self, layer_name: str) -> str:
        if layer_name in self._ffn_layers:
            return layer_name
        if layer_name in self._name_map:
            return self._name_map[layer_name]
        raise KeyError(f"Layer '{layer_name}' not found")

    def get_weight(self, layer_name: str) -> torch.Tensor:
        return self._ffn_layers[self._resolve_name(layer_name)].weight.data.float()

    def get_bias(self, layer_name: str) -> torch.Tensor | None:
        m = self._ffn_layers[self._resolve_name(layer_name)]
        return m.bias.data.float() if m.bias is not None else None

    def get_gradient(self, layer_name: str) -> torch.Tensor | None:
        w = self._ffn_layers[self._resolve_name(layer_name)].weight
        return w.grad.float() if w.grad is not None else None

    def layer_names(self) -> list[str]:
        return list(self._ffn_layers.keys())

    def get_ffn_layer_names(self) -> list[str]:
        """Returns FFN layer names filtered by patterns. Used by orchestration."""
        from domain.constants import is_ffn_layer
        all_names = list(dict(self._model.named_modules()).keys())
        return [n for n in all_names if is_ffn_layer(n)]

    def forward_backward(self, batch: torch.Tensor) -> float:
        self.model.zero_grad()
        loss = self.model(batch, labels=batch).loss
        loss.backward()
        return loss.item()

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def save_ffn_weights(self) -> dict:
        """Save a deep copy of all FFN weights for later restoration."""
        import torch
        return {
            name: self.get_weight(name).clone()
            for name in self.layer_names()
        }

    def restore_ffn_weights(self, saved: dict) -> None:
        """Restore FFN weights from a previously saved copy."""
        for name, w in saved.items():
            self.get_weight(name).data.copy_(w)
