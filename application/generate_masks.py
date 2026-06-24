"""Use Case : Generate binary masks from scores."""

from dataclasses import dataclass
import torch
from domain.scoring.masks import apply_percentile_mask, count_sparsity
from domain.scoring.ports import ScorePersister, MaskPersister


@dataclass
class MaskResult:
    masks: dict[str, torch.Tensor]
    sparsity: dict
    keep_fraction: float


class GenerateMasksUseCase:
    """Generates binary masks from importance scores."""

    def __init__(
        self,
        score_persister: ScorePersister,
        mask_persister: MaskPersister,
    ):
        self.score_persister = score_persister
        self.mask_persister = mask_persister

    def execute(
        self,
        scores_path: str,
        output_path: str,
        keep_fraction: float = 0.3,
    ) -> MaskResult:
        scores = self.score_persister.load(scores_path)
        masks = {}
        for name, score in scores.items():
            masks[name] = apply_percentile_mask(score, keep_fraction)

        self.mask_persister.save(masks, output_path)
        sparsity = count_sparsity(masks)

        return MaskResult(
            masks=masks,
            sparsity=sparsity,
            keep_fraction=keep_fraction,
        )
