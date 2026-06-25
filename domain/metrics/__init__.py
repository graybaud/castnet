"""Domain Metrics — Pure evaluation functions."""

from domain.metrics.perplexity import (
    compute_perplexity_from_loss,
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
from domain.metrics.benchmark import (
    apply_magnitude_pruning,
    apply_magnitude_pruning_all_layers,
    compute_magnitude_benchmark,
)
from domain.metrics.rapid_adapt import compute_adaptation_rate
from domain.metrics.noise_robust import compute_noise_degradation
from domain.metrics.relearn_rate import compute_weight_change
from domain.metrics.lm_eval_tasks import (
    TASK_REGISTRY,
    get_task_config,
    list_available_tasks,
    compute_mmlu_summary,
)
from domain.metrics.neuron_freq import (
    compute_activation_frequency,
    compute_activation_frequency_all_layers,
)
from domain.metrics.custom_domain import (
    evaluate_custom_domain,
    CUSTOM_DOMAIN_QUESTIONS,
)

__all__ = [
    "compute_perplexity_from_loss",
    "compute_perplexity_from_total",
    "run_greedy_tests",
    "run_sampled_tests",
    "analyze_repetition",
    "QUALITATIVE_PROMPTS",
    "count_dead_neurons",
    "count_dead_neurons_all_layers",
    "analyze_weight_distribution",
    "compute_kl_divergence",
    "compute_kl_divergence_batched",
    "topk_overlap",
    "topk_overlap_batched",
    "compute_stability",
    "DEFAULT_STABILITY_PROMPTS",
    "apply_magnitude_pruning",
    "apply_magnitude_pruning_all_layers",
    "compute_magnitude_benchmark",
    "compute_adaptation_rate",
    "compute_noise_degradation",
    "compute_weight_change",
    "TASK_REGISTRY",
    "get_task_config",
    "list_available_tasks",
    "compute_mmlu_summary",
    "compute_activation_frequency",
    "compute_activation_frequency_all_layers",
    "evaluate_custom_domain",
    "CUSTOM_DOMAIN_QUESTIONS",
]
