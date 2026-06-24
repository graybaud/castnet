"""Use Case : Extraction des scores d'importance."""

from dataclasses import dataclass
import torch
from domain.scoring.ports import (
    WeightProvider, BatchProvider, ScorePersister,
)
from domain.scoring.strategies import ScoringStrategy
from infrastructure.hooks.activation_collector import ActivationCollector


@dataclass
class ExtractResult:
    scores: dict[str, torch.Tensor]
    num_layers: int
    num_batches: int


class ExtractScoresUseCase:
    """
    Cas d'usage : Extraire les scores d'importance.

    Coordonne :
    - WeightProvider (infra) pour les poids/gradients
    - BatchProvider (infra) pour les donnees
    - ActivationCollector (infra) pour les activations
    - ScoringStrategy (metier) pour le calcul
    - ScorePersister (infra) pour la sauvegarde
    """

    def __init__(
        self,
        strategy: ScoringStrategy,
        model: WeightProvider,
        dataset: BatchProvider,
        collector: ActivationCollector,
        persister: ScorePersister,
    ):
        self.strategy = strategy
        self.model = model
        self.dataset = dataset
        self.collector = collector
        self.persister = persister

    def execute(self, num_batches: int, output_path: str) -> ExtractResult:
        self.collector.attach()

        accumulated = {
            name: torch.zeros_like(self.model.get_weight(name))
            for name in self.model.layer_names()
        }

        for batch_idx, batch in enumerate(self.dataset.get_batches(num_batches)):
            batch = batch.to(self.model.get_weight(self.model.layer_names()[0]).device)
            self.model.forward_backward(batch)

            for layer_name in self.model.layer_names():
                try:
                    scores = self.strategy.calculate(
                        self.model, self.collector, layer_name
                    )
                    accumulated[layer_name] += scores.cpu()
                except Exception as e:
                    print(f"  Warning: {layer_name} — {e}")

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}")

        self.collector.detach()

        for name in accumulated:
            s = accumulated[name] / num_batches
            s_max = s.max()
            if s_max > 0:
                accumulated[name] = s / s_max

        self.persister.save(accumulated, output_path)

        return ExtractResult(
            scores=accumulated,
            num_layers=len(accumulated),
            num_batches=num_batches,
        )
