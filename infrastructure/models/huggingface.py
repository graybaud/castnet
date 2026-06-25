"""Adaptateur HuggingFace — Implemente WeightProvider."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from domain.scoring.ports import WeightProvider


class HuggingFaceWeightProvider(WeightProvider):
    """Fournit poids et gradients depuis un modele HuggingFace."""

    FFN_PATTERNS = [
        "fc1", "fc2", "c_fc", "c_proj",
        "gate_proj", "up_proj", "down_proj",
        "mlp.fc1", "mlp.fc2", "mlp.c_fc", "mlp.c_proj",
    ]

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if "cuda" in device else None,
        )
        self._device = device
        self._ffn_layers = self._detect_ffn_layers()
        self._name_map = {n.replace(".", "_"): n for n in self._ffn_layers}

    @property
    def _model(self):
        return self.model

    def _detect_ffn_layers(self) -> dict[str, nn.Module]:
        layers = {}
        for name, module in self.model.named_modules():
            # Accepter nn.Linear ET Conv1D (vieux GPT-2)
            if not isinstance(module, (nn.Linear,)):
                # Verifier si c'est un Conv1D (pas dans nn, mais dans GPT2)
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
        return [n.replace(".", "_") for n in self._ffn_layers.keys()]

    def forward_backward(self, batch: torch.Tensor) -> float:
        self.model.train()
        loss = self.model(batch, labels=batch).loss
        loss.backward()
        return loss.item()

    def zero_grad(self) -> None:
        self.model.zero_grad()
