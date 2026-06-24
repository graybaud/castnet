"""Bridge between CastNet domain and apl-pruning formulas."""

import torch
import numpy as np
from apl_pruning import MiniAPLParser
from domain.scoring.ports import WeightProvider, ActivationProvider


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy float32."""
    return tensor.detach().float().cpu().numpy()


def _to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np.asarray(array)).float()


class APLScoringBridge:
    """Evaluates APL formulas against WeightProvider and ActivationProvider."""

    def __init__(self, formula: str):
        self.parser = MiniAPLParser()
        self.formula = formula

    def calculate(
        self,
        weights: WeightProvider,
        activations: ActivationProvider,
        layer_name: str,
    ) -> torch.Tensor:
        """Extract variables from providers, evaluate APL formula."""
        W = _to_numpy(weights.get_weight(layer_name))
        X_in = _to_numpy(activations.get_input_activations(layer_name))
        X_out = _to_numpy(activations.get_output_activations(layer_name))
        grad = weights.get_gradient(layer_name)
        grad_np = _to_numpy(grad) if grad is not None else None

        variables = {"W": W, "act": X_in, "act_out": X_out}
        if grad_np is not None:
            variables["grad"] = grad_np

        self.parser.set_variables(**variables)
        result = self.parser.evaluate(self.formula)
        return _to_torch(result)
