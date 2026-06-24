"""Orchestration — Score extraction entry point."""

import hydra
from omegaconf import DictConfig
import torch

from domain.scoring.strategies import get_strategy
from infrastructure.models.huggingface import HuggingFaceWeightProvider
from infrastructure.data.providers import WikiTextProvider
from infrastructure.hooks.activation_collector import ActivationCollector
from infrastructure.persistence.safetensors_persister import SafetensorsScorePersister
from application.extract_scores import ExtractScoresUseCase


@hydra.main(config_path="../configs", config_name="extract", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {cfg.model} | Method: {cfg.method}")

    # Infrastructure
    model = HuggingFaceWeightProvider(cfg.model, device)
    dataset = WikiTextProvider(cfg.model, cfg.max_len)

    real_names = list(dict(model._model.named_modules()).keys())
    ffn_names = [
        n for n in real_names
        if any(p in n.lower() for p in HuggingFaceWeightProvider.FFN_PATTERNS)
    ]
    collector = ActivationCollector(model._model, ffn_names)
    persister = SafetensorsScorePersister()

    # Domain
    strategy = get_strategy(cfg.method)

    # Application
    use_case = ExtractScoresUseCase(
        strategy=strategy,
        model=model,
        dataset=dataset,
        collector=collector,
        persister=persister,
    )

    # Execute
    result = use_case.execute(
        num_batches=cfg.num_batches,
        output_path=cfg.output,
    )

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Model   : {cfg.model}")
    print(f"  Method  : {cfg.method}")
    print(f"  Layers  : {result.num_layers}")
    print(f"  Batches : {result.num_batches}")
    print(f"  Output  : {cfg.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
