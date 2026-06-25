"""Use Cases : Model evaluation."""

from dataclasses import dataclass
import torch
from domain.metrics.perplexity import compute_perplexity_from_total
from domain.scoring.ports import WeightProvider, MaskPersister
from domain.scoring.masks import count_sparsity, apply_masks_to_weights
from infrastructure.data.perplexity_dataset import WikiTextValidationDataset


@dataclass
class PerplexityResult:
    perplexity: float
    total_tokens: int


@dataclass
class SparsityResult:
    sparsity: dict


@dataclass
class EvalResult:
    perplexity: float
    sparsity: dict
    total_tokens: int


class EvaluatePerplexityUseCase:
    """Evaluate perplexity on WikiText-2 validation (contiguous blocks)."""

    def __init__(
        self,
        model,
        tokenizer_name: str = "facebook/opt-125m",
        block_size: int = 512,
        max_tokens: int = 50000,
    ):
        self.model = model
        self.dataset = WikiTextValidationDataset(tokenizer_name, block_size, max_tokens)

    def execute(self, device: str = "cpu") -> PerplexityResult:
        was_training = self.model.training
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        with torch.no_grad():
            for block in self.dataset.get_blocks(device):
                out = self.model(block, labels=block)
                total_loss += out.loss.item() * block.size(1)
                total_tok += block.size(1)
        if was_training:
            self.model.train()
        return PerplexityResult(
            perplexity=compute_perplexity_from_total(total_loss, total_tok),
            total_tokens=total_tok,
        )


class EvaluateSparsityUseCase:
    """Measure sparsity from a mask file."""

    def __init__(self, mask_persister: MaskPersister):
        self.mask_persister = mask_persister

    def execute(self, mask_path: str) -> SparsityResult:
        masks = self.mask_persister.load(mask_path)
        return SparsityResult(sparsity=count_sparsity(masks))


class EvaluateModelUseCase:
    """Evaluate both perplexity and sparsity for a masked model."""

    def __init__(
        self,
        model: WeightProvider,
        mask_persister: MaskPersister,
        tokenizer_name: str = "facebook/opt-125m",
    ):
        self.model = model
        self.mask_persister = mask_persister
        self.perp_uc = EvaluatePerplexityUseCase(model, tokenizer_name)
        self.sp_uc = EvaluateSparsityUseCase(mask_persister)
        self._dense_weights = {
            name: model.get_weight(name).clone()
            for name in model.layer_names()
        }

    def execute(self, mask_path: str, device: str = "cpu") -> EvalResult:
        # Restore + apply mask
        for name, w in self._dense_weights.items():
            self.model.get_weight(name).data.copy_(w)
        masks = self.mask_persister.load(mask_path)
        apply_masks_to_weights(masks, self.model)

        # Evaluate
        perp_result = self.perp_uc.execute(device)
        sp_result = self.sp_uc.execute(mask_path)

        return EvalResult(
            perplexity=perp_result.perplexity,
            sparsity=sp_result.sparsity,
            total_tokens=perp_result.total_tokens,
        )
