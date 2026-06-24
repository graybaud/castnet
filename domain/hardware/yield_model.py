"""Yield model — Murphy, Seeds, Negative Binomial — Pure domain."""

import math
from domain.hardware.constants import DEFECT_DENSITY, CLUSTERING_ALPHA


def murphy_yield(area_mm2: float, D0: float) -> float:
    """Murphy yield model."""
    D0A = D0 * (area_mm2 / 100.0)
    if D0A < 1e-8:
        return 1.0
    return (1.0 - math.exp(-D0A)) / D0A


def seeds_yield(area_mm2: float, D0: float) -> float:
    """Seeds yield model."""
    return math.exp(-D0 * (area_mm2 / 100.0))


def negative_binomial_yield(area_mm2: float, D0: float, alpha: float = 1.0) -> float:
    """Negative binomial yield model."""
    D0A = D0 * (area_mm2 / 100.0)
    if alpha <= 0:
        return seeds_yield(area_mm2, D0)
    return (1.0 + D0A / alpha) ** (-alpha)


def chips_per_wafer(side_mm: float, wafer_diam: float = 200.0, edge_loss: float = 5.0) -> int:
    """Estimate gross dies per wafer."""
    R_eff = wafer_diam / 2.0 - edge_loss
    A = side_mm ** 2
    if A <= 0 or side_mm > wafer_diam:
        return 0
    return max(0, int(math.pi * R_eff**2 / A - math.pi * R_eff / math.sqrt(2 * A)))


def estimate_yield(area_mm2: float, side_mm: float, process: str = "130nm") -> dict:
    """Full yield estimation for a given process node."""
    D0 = DEFECT_DENSITY.get(process, 0.3)
    alpha = CLUSTERING_ALPHA.get(process, 1.0)
    
    y_nb = negative_binomial_yield(area_mm2, D0, alpha)
    gross = chips_per_wafer(side_mm)
    working = int(gross * y_nb)
    cost = 2000.0 / working if working > 0 else float("inf")
    
    return {
        "process": process,
        "defect_density_per_cm2": D0,
        "clustering_alpha": alpha,
        "chip_area_mm2": round(area_mm2, 2),
        "chip_side_mm": round(side_mm, 2),
        "yield_neg_binomial_pct": round(y_nb * 100, 1),
        "gross_dies_per_wafer": gross,
        "working_dies_per_wafer": working,
        "cost_per_working_die_usd": round(cost, 2),
    }
