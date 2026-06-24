"""Collecteur d'activations — Centralise TOUTE la logique de hooks."""

import torch
import torch.nn as nn
from domain.scoring.ports import ActivationProvider


class ActivationCollector(ActivationProvider):
    """Collecte les activations via hooks PyTorch. Usage unique pour tout le projet."""

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self._model = model
        self._layer_names = layer_names
        self._hooks: list = []
        self._inputs: dict[str, torch.Tensor] = {}
        self._outputs: dict[str, torch.Tensor] = {}

    def attach(self) -> None:
        modules = dict(self._model.named_modules())
        for name in self._layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            x_in = input[0].detach()
            x_out = output.detach()
            if x_in.dim() == 3:
                x_in = x_in.reshape(-1, x_in.shape[-1])
            if x_out.dim() == 3:
                x_out = x_out.reshape(-1, x_out.shape[-1])
            self._inputs[name] = x_in
            self._outputs[name] = x_out
        return hook_fn

    def get_input_activations(self, layer_name: str) -> torch.Tensor:
        return self._inputs.get(layer_name)

    def get_output_activations(self, layer_name: str) -> torch.Tensor:
        return self._outputs.get(layer_name)
