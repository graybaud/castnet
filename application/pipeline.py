"""Use Case : Full CastNet pipeline."""

from dataclasses import dataclass
from application.extract_scores import ExtractScoresUseCase, ExtractResult
from application.generate_masks import GenerateMasksUseCase, MaskResult
from application.finetune import FinetuneUseCase, FinetuneResult
from application.evaluate import EvaluateUseCase, EvalResult


@dataclass
class PipelineResult:
    extract: ExtractResult
    mask: MaskResult
    finetune: FinetuneResult | None
    evaluate: EvalResult | None


class CastNetPipeline:
    """Orchestrates the full CastNet workflow."""

    def __init__(
        self,
        extract_uc: ExtractScoresUseCase,
        mask_uc: GenerateMasksUseCase,
        finetune_uc: FinetuneUseCase | None = None,
        evaluate_uc: EvaluateUseCase | None = None,
    ):
        self.extract_uc = extract_uc
        self.mask_uc = mask_uc
        self.finetune_uc = finetune_uc
        self.evaluate_uc = evaluate_uc

    def run(
        self,
        scores_path: str,
        masks_path: str,
        checkpoint_path: str,
        num_batches: int = 200,
        keep_fraction: float = 0.3,
        epochs: int = 5,
        eval_batches: int = 50,
    ) -> PipelineResult:
        print("[1/4] Extracting scores...")
        extract_result = self.extract_uc.execute(num_batches, scores_path)

        print("[2/4] Generating masks...")
        mask_result = self.mask_uc.execute(scores_path, masks_path, keep_fraction)

        finetune_result = None
        if self.finetune_uc:
            print("[3/4] Fine-tuning...")
            finetune_result = self.finetune_uc.execute(
                masks_path, checkpoint_path, epochs=epochs
            )

        eval_result = None
        if self.evaluate_uc:
            print("[4/4] Evaluating...")
            eval_result = self.evaluate_uc.execute(masks_path, eval_batches)

        return PipelineResult(
            extract=extract_result,
            mask=mask_result,
            finetune=finetune_result,
            evaluate=eval_result,
        )
