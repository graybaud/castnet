"""Activation collector — accumulates activations across batches (like legacy)."""

import torch
import torch.nn as nn
from domain.scoring.ports import ActivationProvider


class ActivationCollector(ActivationProvider):
    """Collects activations via PyTorch hooks. Accumulates across batches."""

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self._model = model
        self._layer_names = layer_names
        self._hooks: list = []
        self._inputs: dict[str, torch.Tensor] = {}
        self._outputs: dict[str, torch.Tensor] = {}
        self._input_list: dict[str, list] = {}
        self._output_list: dict[str, list] = {}
        self._num_batches: int = 0

    def attach(self) -> None:
        modules = dict(self._model.named_modules())
        self._input_list = {name: [] for name in self._layer_names}
        self._output_list = {name: [] for name in self._layer_names}
        for name in self._layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
            alt_name = name.replace(".", "_")
            if alt_name != name and alt_name in modules:
                hook = modules[alt_name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset_accumulators(self) -> None:
        self._input_list = {name: [] for name in self._layer_names}
        self._output_list = {name: [] for name in self._layer_names}
        self._inputs.clear()
        self._outputs.clear()
        self._num_batches = 0

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            try:
                x_in = input[0].detach().float()
                x_out = output.detach().float()
                if x_in.dim() == 3:
                    x_in = x_in.reshape(-1, x_in.shape[-1])
                if x_out.dim() == 3:
                    x_out = x_out.reshape(-1, x_out.shape[-1])
                self._inputs[name] = x_in
                self._outputs[name] = x_out
                if name in self._input_list:
                    self._input_list[name].append(x_in.cpu())
                    self._output_list[name].append(x_out.cpu())
                self._num_batches = max(
                    len(self._input_list.get(name, [])),
                    self._num_batches,
                )
            except Exception:
                pass
        return hook_fn

    def get_input_activations(self, layer_name: str) -> torch.Tensor:
        return self._inputs.get(layer_name)

    def get_output_activations(self, layer_name: str) -> torch.Tensor:
        return self._outputs.get(layer_name)

    def get_input_concatenated(self, layer_name: str, max_tokens: int = None):
        lst = self._input_list.get(layer_name, [])
        if not lst:
            alt = layer_name.replace(".", "_")
            lst = self._input_list.get(alt, [])
        if not lst:
            return None
        result = torch.cat(lst, dim=0)
        if max_tokens and result.shape[0] > max_tokens:
            result = result[:max_tokens]
        return result

    def get_output_concatenated(self, layer_name: str, max_tokens: int = None):
        lst = self._output_list.get(layer_name, [])
        if not lst:
            alt = layer_name.replace(".", "_")
            lst = self._output_list.get(alt, [])
        if not lst:
            return None
        result = torch.cat(lst, dim=0)
        if max_tokens and result.shape[0] > max_tokens:
            result = result[:max_tokens]
        return result

    @property
    def num_batches(self) -> int:
        return self._num_batches
