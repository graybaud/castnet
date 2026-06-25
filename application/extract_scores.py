"""Use Case : Extract importance scores."""

from dataclasses import dataclass
import torch
import time
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
    """Extract importance scores from a model."""

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
        layer_names = self.model.layer_names()
        needs_grad = self._needs_gradient()
        
        print(f"[execute] {len(layer_names)} layers, needs_grad={needs_grad}")
        
        self.collector.attach()

        accumulated = {
            name: torch.zeros_like(self.model.get_weight(name))
            for name in layer_names
        }

        first_layer = layer_names[0]
        device = self.model.get_weight(first_layer).device
        print(f"[execute] Device: {device}")

        t_total_start = time.time()

        for batch_idx, batch in enumerate(self.dataset.get_batches(num_batches)):
            t_batch_start = time.time()
            print(f"[execute] Batch {batch_idx+1}/{num_batches}")
            
            batch = batch.to(device)
            
            if needs_grad:
                t_fw = time.time()
                self.model.forward_backward(batch)
                print(f"[execute]   Forward+backward: {time.time()-t_fw:.1f}s")
            else:
                t_fw = time.time()
                with torch.no_grad():
                    self.model.model(batch)
                print(f"[execute]   Forward only: {time.time()-t_fw:.1f}s")

            for layer_name in layer_names:
                try:
                    scores = self.strategy.calculate(
                        self.model, self.collector, layer_name
                    )
                    accumulated[layer_name] += scores.cpu()
                except Exception as e:
                    print(f"[execute]   {layer_name}: ERROR — {e}")

            print(f"[execute] Batch {batch_idx+1}/{num_batches} done in {time.time()-t_batch_start:.1f}s")

        print(f"[execute] All batches done in {time.time()-t_total_start:.1f}s")
        self.collector.detach()

        print(f"[execute] Normalizing...")
        for name in accumulated:
            s = accumulated[name] / num_batches
            s_max = s.max()
            if s_max > 0:
                accumulated[name] = s / s_max

        print(f"[execute] Saving to {output_path}...")
        self.persister.save(accumulated, output_path)

        return ExtractResult(
            scores=accumulated,
            num_layers=len(accumulated),
            num_batches=num_batches,
        )

    def _needs_gradient(self) -> bool:
        """Check if the strategy requires gradients."""
        return hasattr(self.strategy, 'needs_grad') and self.strategy.needs_grad
