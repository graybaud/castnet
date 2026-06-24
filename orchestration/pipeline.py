"""Orchestration — Full CastNet pipeline."""

import hydra
from omegaconf import DictConfig
import torch

from domain.scoring.strategies import get_strategy
from infrastructure.models.huggingface import HuggingFaceWeightProvider
from infrastructure.data.providers import WikiTextProvider
from infrastructure.hooks.activation_collector import ActivationCollector
from infrastructure.persistence.safetensors_persister import (
    SafetensorsScorePersister,
    SafetensorsMaskPersister,
)
from application.extract_scores import ExtractScoresUseCase
from application.generate_masks import GenerateMasksUseCase
from application.finetune import FinetuneUseCase
from application.evaluate import EvaluateUseCase
from application.pipeline import CastNetPipeline


@hydra.main(config_path="../configs", config_name="pipeline", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {cfg.model}")

    # Shared infrastructure
    model = HuggingFaceWeightProvider(cfg.model, device)
    dataset = WikiTextProvider(cfg.model, cfg.max_len)

    real_names = list(dict(model._model.named_modules()).keys())
    ffn_names = [
        n for n in real_names
        if any(p in n.lower() for p in HuggingFaceWeightProvider.FFN_PATTERNS)
    ]
    collector = ActivationCollector(model._model, ffn_names)
    score_persister = SafetensorsScorePersister()
    mask_persister = SafetensorsMaskPersister()

    # Use cases
    strategy = get_strategy(cfg.extract.method)

    extract_uc = ExtractScoresUseCase(strategy, model, dataset, collector, score_persister)
    mask_uc = GenerateMasksUseCase(score_persister, mask_persister)
    finetune_uc = FinetuneUseCase(model, dataset, mask_persister) if cfg.run_finetune else None
    evaluate_uc = EvaluateUseCase(model, dataset, mask_persister) if cfg.run_evaluate else None

    pipeline = CastNetPipeline(extract_uc, mask_uc, finetune_uc, evaluate_uc)

    result = pipeline.run(
        scores_path=cfg.scores_path,
        masks_path=cfg.masks_path,
        checkpoint_path=cfg.checkpoint_path,
        num_batches=cfg.extract.num_batches,
        keep_fraction=cfg.masks.keep_fraction,
        epochs=cfg.finetune.epochs,
    )

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    if result.evaluate:
        print(f"  Perplexity : {result.evaluate.perplexity:.1f}")
        print(f"  Sparsity   : {result.evaluate.sparsity['overall_sparsity_pct']}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
