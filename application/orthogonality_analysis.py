"""Use Case : Orthogonality heatmap between scoring methods."""

from dataclasses import dataclass
import numpy as np
from domain.scoring.masks import apply_percentile_mask
from domain.analysis.correlation import mask_overlap, orthogonality_score


@dataclass
class OrthogonalityResult:
    methods: list[str]
    keep_fraction: float
    overlap_matrix: list[list[float]]
    average_overlap: float
    orthogonality_score: float
    most_orthogonal_pairs: list[tuple[str, str, float]]


class OrthogonalityAnalysisUseCase:
    """Compute overlap matrix between masks from different scoring methods."""

    def __init__(self, score_persister):
        self.score_persister = score_persister

    def execute(
        self,
        method_paths: dict[str, str],
        keep_fraction: float = 0.30,
    ) -> OrthogonalityResult:
        # Load and normalize scores
        available = {}
        for name, path in method_paths.items():
            try:
                scores = self.score_persister.load(path)
                for n in scores:
                    s = scores[n]
                    s_max = s.max()
                    if s_max > 0:
                        scores[n] = s / s_max
                available[name] = scores
            except FileNotFoundError:
                pass

        if len(available) < 2:
            raise ValueError("Need at least 2 methods to compare")

        names = list(available.keys())
        n = len(names)

        # Generate masks
        masks = {}
        for name, scores in available.items():
            masks[name] = {
                layer: apply_percentile_mask(scores[layer], keep_fraction)
                for layer in scores
            }

        # Build overlap matrix
        matrix = np.zeros((n, n))
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = mask_overlap(masks[name_i], masks[name_j])

        # Compute statistics
        off_diag = [
            matrix[i, j] for i in range(n) for j in range(i + 1, n)
        ]
        avg_overlap = float(np.mean(off_diag)) if off_diag else 0.0
        ortho_score = orthogonality_score([float(x) for x in off_diag])

        # Find most orthogonal pairs
        pairs = [
            (names[i], names[j], float(matrix[i, j]))
            for i in range(n) for j in range(i + 1, n)
        ]
        pairs.sort(key=lambda x: x[2])

        return OrthogonalityResult(
            methods=names,
            keep_fraction=keep_fraction,
            overlap_matrix=matrix.tolist(),
            average_overlap=round(avg_overlap, 4),
            orthogonality_score=round(ortho_score, 1),
            most_orthogonal_pairs=pairs[:3],
        )
