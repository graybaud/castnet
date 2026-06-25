"""Use Case : Sparsity sweep — evaluate perplexity at multiple keep fractions."""

from dataclasses import dataclass
import torch
from domain.scoring.ports import WeightProvider, BatchProvider, ScorePersister, MaskPersister
from domain.scoring.masks import apply_percentile_mask, count_sparsity, apply_masks_to_weights
from domain.metrics.perplexity import compute_perplexity_from_total


@dataclass
class SweepResult:
    method: str
    results: list[dict]  # [{"keep_fraction": 0.3, "sparsity_pct": 70.0, "perplexity": 35.2}, ...]


class SparsitySweepUseCase:
    """Evaluates perplexity at multiple keep fractions for a given scoring method."""

    KEEP_FRACTIONS = [0.90, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]

    def __init__(
        self,
        model: WeightProvider,
        dataset: BatchProvider,
        score_persister: ScorePersister,
        mask_persister: MaskPersister,
    ):
        self.model = model
        self.dataset = dataset
        self.score_persister = score_persister
        self.mask_persister = mask_persister

    def execute(self, scores_path: str, eval_batches: int = 50) -> SweepResult:
        scores = self.score_persister.load(scores_path)
        results = []

        for keep_frac in self.KEEP_FRACTIONS:
            masks = {}
            for name, score in scores.items():
                masks[name] = apply_percentile_mask(score, keep_frac)

            apply_masks_to_weights(masks, self.model)

            total_loss = 0.0
            total_tokens = 0
            for batch in self.dataset.get_batches(eval_batches):
                loss = self.model.forward_backward(batch)
                total_loss += loss * batch.numel()
                total_tokens += batch.numel()

            perplexity = compute_perplexity_from_total(total_loss, total_tokens)
            sparsity = count_sparsity(masks)

            results.append({
                "keep_fraction": keep_frac,
                "sparsity_pct": sparsity["overall_sparsity_pct"],
                "perplexity": round(perplexity, 1),
            })

            print(f"  keep={keep_frac:.2f}  sparsity={sparsity['overall_sparsity_pct']:.1f}%  perp={perplexity:.1f}")

            # Restaurer les poids (recharger le modele)
            # Note: pour l'instant on recrée les masques a chaque iteration

        return SweepResult(method=scores_path, results=results)
