"""CastNet Scoring Infrastructure — All accumulators + APL Bridge (lazy)."""

from infrastructure.scoring.accumulators import (
    accumulate_gradient_scores,
    accumulate_wanda_scores,
    accumulate_magnitude_scores,
    accumulate_chain_scores,
    accumulate_wanda_chain_scores,
    accumulate_softmax_gradient_scores,
    accumulate_weighted_softmax_scores,
    accumulate_sparsegps_scores,
    accumulate_sparsegpt_scores,
    accumulate_gcs_scores,
    accumulate_trident_scores,
)

__all__ = [
    "accumulate_gradient_scores",
    "accumulate_wanda_scores",
    "accumulate_magnitude_scores",
    "accumulate_chain_scores",
    "accumulate_wanda_chain_scores",
    "accumulate_softmax_gradient_scores",
    "accumulate_weighted_softmax_scores",
    "accumulate_sparsegps_scores",
    "accumulate_sparsegpt_scores",
    "accumulate_gcs_scores",
    "accumulate_trident_scores",
    "get_apl_bridge",
]


def get_apl_bridge():
    """Lazy import of APLScoringBridge."""
    from infrastructure.scoring.apl_bridge import APLScoringBridge
    return APLScoringBridge
