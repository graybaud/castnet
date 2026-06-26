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
        self._input_accum: dict[str, torch.Tensor] = {}
        self._output_accum: dict[str, torch.Tensor] = {}
        self._batch_count: dict[str, int] = {}
        self._num_batches: int = 0

    def attach(self) -> None:
        modules = dict(self._model.named_modules())
        for name in self._layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
            alt_name = name.replace(".", "_")
            if alt_name != name and alt_name in modules:
                hook = modules[alt_name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
        # Initialize list accumulators (legacy-style: store each batch)
        self._input_list: dict[str, list] = {name: [] for name in self._layer_names}
        self._output_list: dict[str, list] = {name: [] for name in self._layer_names}

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset_accumulators(self) -> None:
        """Reset accumulated activations for a new scoring run."""
        self._input_accum.clear()
        self._output_accum.clear()
        self._batch_count.clear()
        self._num_batches = 0
        if hasattr(self, '_input_list'):
            self._input_list.clear()
            for name in self._layer_names:
                self._input_list[name] = []
        if hasattr(self, '_output_list'):
            self._output_list.clear()
            for name in self._layer_names:
                self._output_list[name] = []

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            x_in = input[0].detach().float()
            x_out = output.detach().float()
            if x_in.dim() == 3:
                x_in = x_in.reshape(-1, x_in.shape[-1])
            if x_out.dim() == 3:
                x_out = x_out.reshape(-1, x_out.shape[-1])
            # Accumulate (sum)
            if name not in self._input_accum:
                self._input_accum[name] = x_in
                self._output_accum[name] = x_out
                self._batch_count[name] = 1
            else:
                self._input_accum[name] = self._input_accum[name] + x_in
                self._output_accum[name] = self._output_accum[name] + x_out
                self._batch_count[name] += 1
            # Accumulate (list — for GPS-style concat)
            if hasattr(self, '_input_list'):
                self._input_list[name].append(x_in.cpu())
            if hasattr(self, '_output_list'):
                self._output_list[name].append(x_out.cpu())
            # Also store latest for legacy compatibility
            self._inputs[name] = x_in
            self._outputs[name] = x_out
            self._num_batches = max(self._batch_count.values()) if self._batch_count else 0
        return hook_fn

    def get_input_activations(self, layer_name: str) -> torch.Tensor:
        """Return accumulated input activations (sum over all batches)."""
        result = self._input_accum.get(layer_name)
        if result is None:
            alt_name = layer_name.replace(".", "_")
            result = self._input_accum.get(alt_name)
        return result

    def get_output_activations(self, layer_name: str) -> torch.Tensor:
        """Return accumulated output activations (sum over all batches)."""
        result = self._output_accum.get(layer_name)
        if result is None:
            alt_name = layer_name.replace(".", "_")
            result = self._output_accum.get(alt_name)
        return result

    def get_input_mean(self, layer_name: str) -> torch.Tensor:
        """Return mean input activation per feature (like legacy Wanda)."""
        acc = self.get_input_activations(layer_name)
        if acc is None:
            return None
        n = self._batch_count.get(layer_name, self._num_batches)
        return acc / max(n, 1)

    def get_output_mean(self, layer_name: str) -> torch.Tensor:
        """Return mean output activation."""
        acc = self.get_output_activations(layer_name)
        if acc is None:
            return None
        n = self._batch_count.get(layer_name, self._num_batches)
        return acc / max(n, 1)

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def get_input_concatenated(self, layer_name: str, max_tokens: int = None):
        """Return all input activations concatenated (legacy GPS-style).
        
        Returns a tensor [N_total, d_in] by concatenating all batches.
        """
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
        """Return all output activations concatenated (legacy GPS-style)."""
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
