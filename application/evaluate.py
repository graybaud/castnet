"""Use Case : Evaluate a sparse model."""

from dataclasses import dataclass
import torch
from domain.metrics.perplexity import compute_perplexity_from_total
from domain.scoring.ports import WeightProvider, BatchProvider
from domain.scoring.masks import count_sparsity
from domain.scoring.ports import MaskPersister


@dataclass
class EvalResult:
    perplexity: float
    sparsity: dict


class EvaluateUseCase:
    """Evaluates perplexity and sparsity of a sparse model."""

    def __init__(
        self,
        model: WeightProvider,
        dataset: BatchProvider,
        mask_persister: MaskPersister,
    ):
        self.model = model
        self.dataset = dataset
        self.mask_persister = mask_persister

    def execute(
        self,
        mask_path: str,
        num_batches: int = 50,
    ) -> EvalResult:
        masks = self.mask_persister.load(mask_path)

        total_loss = 0.0
        total_tokens = 0
        for batch in self.dataset.get_batches(num_batches):
            loss = self.model.forward_backward(batch)
            total_loss += loss * batch.numel()
            total_tokens += batch.numel()

        perplexity = compute_perplexity_from_total(total_loss, total_tokens)
        sparsity = count_sparsity(masks)

        return EvalResult(
            perplexity=perplexity,
            sparsity=sparsity,
        )
