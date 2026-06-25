"""Use Case : Correlation analysis between scoring methods."""

from dataclasses import dataclass
import torch
import numpy as np
from domain.analysis.correlation import pearson_correlation, interpret_correlation


@dataclass
class CorrelationResult:
    method_a: str
    method_b: str
    per_layer: dict[str, float]
    average: float
    interpretation: str


class CorrelationAnalysisUseCase:
    """Compute Pearson correlation between neuron scores from different methods."""

    def __init__(self, score_persister):
        self.score_persister = score_persister

    def execute(
        self,
        scores_path_a: str,
        scores_path_b: str,
        method_a: str = "A",
        method_b: str = "B",
    ) -> CorrelationResult:
        scores_a = self.score_persister.load(scores_path_a)
        scores_b = self.score_persister.load(scores_path_b)

        # Normalize
        for name in scores_a:
            s = scores_a[name]
            s_max = s.max()
            if s_max > 0:
                scores_a[name] = s / s_max
        for name in scores_b:
            s = scores_b[name]
            s_max = s.max()
            if s_max > 0:
                scores_b[name] = s / s_max

        per_layer = {}
        correlations = []

        for name in sorted(scores_a.keys()):
            if name not in scores_b:
                continue
            a = scores_a[name].mean(dim=1).float().numpy()
            b = scores_b[name].mean(dim=1).float().numpy()
            r = pearson_correlation(a.tolist(), b.tolist())
            per_layer[name] = round(r, 4)
            correlations.append(r)

        avg = np.mean(correlations) if correlations else 0.0
        return CorrelationResult(
            method_a=method_a,
            method_b=method_b,
            per_layer=per_layer,
            average=round(float(avg), 4),
            interpretation=interpret_correlation(avg),
        )
