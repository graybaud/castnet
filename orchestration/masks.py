"""Orchestration — Mask generation entry point."""

import hydra
from omegaconf import DictConfig

from infrastructure.persistence.safetensors_persister import (
    SafetensorsScorePersister,
    SafetensorsMaskPersister,
)
from application.generate_masks import GenerateMasksUseCase


@hydra.main(config_path="../configs", config_name="masks", version_base=None)
def main(cfg: DictConfig):
    print(f"Scores: {cfg.scores_path} | Keep fraction: {cfg.keep_fraction}")

    score_persister = SafetensorsScorePersister()
    mask_persister = SafetensorsMaskPersister()

    use_case = GenerateMasksUseCase(score_persister, mask_persister)
    result = use_case.execute(
        scores_path=cfg.scores_path,
        output_path=cfg.output,
        keep_fraction=cfg.keep_fraction,
    )

    print(f"\n{'=' * 60}")
    print(f"  MASKS GENERATED")
    print(f"  Active : {result.sparsity['total_active']:,} / {result.sparsity['total_connections']:,}")
    print(f"  Sparsity : {result.sparsity['overall_sparsity_pct']}%")
    print(f"  Output : {cfg.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
