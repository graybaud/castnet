"""Domain Scoring — Public API."""

from domain.scoring.ports import (
    WeightProvider,
    ActivationProvider,
    BatchProvider,
    ScorePersister,
    MaskPersister,
)
from domain.scoring.strategies import (
    ScoringStrategy,
    MagnitudeStrategy,
    GradientStrategy,
    WandaStrategy,
    GPSStrategy,
    STRATEGY_REGISTRY,
    get_strategy,
)
from domain.scoring.masks import (
    apply_percentile_mask,
    apply_masks_to_weights,
    count_sparsity,
)

__all__ = [
    # Ports
    "WeightProvider",
    "ActivationProvider",
    "BatchProvider",
    "ScorePersister",
    "MaskPersister",
    # Strategies
    "ScoringStrategy",
    "MagnitudeStrategy",
    "GradientStrategy",
    "WandaStrategy",
    "GPSStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    # Masks
    "apply_percentile_mask",
    "apply_masks_to_weights",
    "count_sparsity",
]
