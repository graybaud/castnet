"""Correlation and orthogonality analysis — Pure domain logic."""

import math
from typing import Optional


def pearson_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def mask_overlap(m1: dict[int, float], m2: dict[int, float]) -> float:
    inter = sum(m1.get(i, 0.0) * m2.get(i, 0.0) for i in set(m1) | set(m2))
    union = sum(m1.values())
    return inter / union if union > 0 else 0.0


def overlap_matrix(
    masks: dict[str, dict[int, float]],
    method_names: Optional[list[str]] = None,
) -> list[list[float]]:
    if method_names is None:
        method_names = sorted(masks.keys())
    n = len(method_names)
    matrix = [[0.0] * n for _ in range(n)]
    for i, ni in enumerate(method_names):
        for j, nj in enumerate(method_names):
            if i == j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = mask_overlap(masks[ni], masks[nj])
    return matrix


def interpret_correlation(r: float) -> str:
    a = abs(r)
    if a < 0.3: return "ORTHOGONAL"
    if a < 0.5: return "PARTIALLY CORRELATED"
    return "HIGHLY CORRELATED"


def orthogonality_score(overlaps: list[float]) -> float:
    if not overlaps: return 100.0
    return (1.0 - sum(overlaps) / len(overlaps)) * 100.0
