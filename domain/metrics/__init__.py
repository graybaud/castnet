"""Domain Metrics — Pure evaluation functions."""

from domain.metrics.perplexity import (
    compute_perplexity_from_loss,
    compute_perplexity_from_logits,
    compute_perplexity_from_total,
)
from domain.metrics.qualitative import (
    run_greedy_tests,
    run_sampled_tests,
    analyze_repetition,
    QUALITATIVE_PROMPTS,
)
from domain.metrics.dead_neurons import (
    count_dead_neurons,
    count_dead_neurons_all_layers,
)
from domain.metrics.weight_distribution import analyze_weight_distribution
from domain.metrics.kl_divergence import (
    compute_kl_divergence,
    compute_kl_divergence_batched,
)
from domain.metrics.topk_rank import (
    topk_overlap,
    topk_overlap_batched,
)
from domain.metrics.stability import (
    compute_stability,
    DEFAULT_STABILITY_PROMPTS,
)

__all__ = [
    # Perplexity
    "compute_perplexity_from_loss",
    "compute_perplexity_from_logits",
    "compute_perplexity_from_total",
    # Qualitative
    "run_greedy_tests",
    "run_sampled_tests",
    "analyze_repetition",
    "QUALITATIVE_PROMPTS",
    # Dead neurons
    "count_dead_neurons",
    "count_dead_neurons_all_layers",
    # Weight distribution
    "analyze_weight_distribution",
    # KL Divergence
    "compute_kl_divergence",
    "compute_kl_divergence_batched",
    # Top-K Rank
    "topk_overlap",
    "topk_overlap_batched",
    # Stability
    "compute_stability",
    "DEFAULT_STABILITY_PROMPTS",
]
