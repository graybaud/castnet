"""Bridge between CastNet domain and apl-pruning formulas."""

import torch
import numpy as np
from apl_pruning import MiniAPLParser
from domain.scoring.ports import WeightProvider, ActivationProvider


def _to_numpy(tensor: torch.Tensor | None) -> np.ndarray | None:
    """Convert torch tensor to numpy float32. Returns None if tensor is None."""
    if tensor is None:
        return None
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
        # Resolve formula name to actual APL expression
        formula = self.formula
        try:
            from domain.scorers import METHODS
            if self.formula in METHODS:
                formula = METHODS[self.formula]["formula"]
        except ImportError:
            pass  # Use formula string directly if METHODS not available
        
        W = _to_numpy(weights.get_weight(layer_name))
        
        X_in = activations.get_input_activations(layer_name)
        X_in_np = _to_numpy(X_in)
        
        X_out = activations.get_output_activations(layer_name)
        X_out_np = _to_numpy(X_out)
        
        grad = weights.get_gradient(layer_name)
        grad_np = _to_numpy(grad)

        variables = {"W": W}
        
        # Optional precomputed metrics
        distortion = getattr(weights, 'get_distortion', None)
        if distortion is not None:
            distortion_np = _to_numpy(distortion(layer_name))
            if distortion_np is not None:
                variables["distortion"] = distortion_np
        if X_in_np is not None:
            variables["act"] = X_in_np
        if X_out_np is not None:
            variables["act_out"] = X_out_np
        if grad_np is not None:
            variables["grad"] = grad_np

        self.parser.set_variables(**variables)
        result = self.parser.evaluate(formula)
        return _to_torch(result)
