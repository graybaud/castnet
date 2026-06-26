"""Use Cases : Advanced evaluation metrics."""

from dataclasses import dataclass
import torch
from domain.scoring.ports import WeightProvider


@dataclass
class MetricResult:
    metric_name: str
    value: dict


class EvaluateDeadNeuronsUseCase:
    """Count dead neurons from a model."""
    
    def __init__(self, model: WeightProvider):
        self.model = model

    def execute(self) -> MetricResult:
        from domain.metrics.dead_neurons import count_dead_neurons
        weights = {name: self.model.get_weight(name) for name in self.model.layer_names()}
        result = {}
        for name, W in weights.items():
            result[name] = count_dead_neurons(W)
        return MetricResult(metric_name="dead_neurons", value=result)


class EvaluateWeightDistributionUseCase:
    """Analyze weight distribution."""
    
    def __init__(self, model: WeightProvider):
        self.model = model

    def execute(self) -> MetricResult:
        from domain.metrics.weight_distribution import analyze_weight_distribution
        result = {}
        for name in self.model.layer_names():
            W = self.model.get_weight(name)
            result[name] = analyze_weight_distribution(W)
        return MetricResult(metric_name="weight_distribution", value=result)


class EvaluateCustomDomainUseCase:
    """Evaluate on custom domain questions."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def execute(self, questions=None) -> MetricResult:
        from domain.metrics.custom_domain import evaluate_custom_domain
        result = evaluate_custom_domain(self.model, self.tokenizer, questions)
        return MetricResult(metric_name="custom_domain", value=result)


class EvaluateNoiseRobustnessUseCase:
    """Evaluate perplexity degradation under noise."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def execute(self, noise_levels=None) -> MetricResult:
        from domain.metrics.noise_robust import compute_noise_degradation
        # Simplified: returns the formula, actual eval needs model forward
        return MetricResult(metric_name="noise_robustness", value={
            "status": "requires model forward with noise injection"
        })
