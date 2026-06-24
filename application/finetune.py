"""Use Case : Frozen-topology fine-tuning."""

from dataclasses import dataclass, field
import torch
from domain.scoring.ports import WeightProvider, BatchProvider, MaskPersister
from domain.scoring.masks import apply_masks_to_weights, count_sparsity


@dataclass
class FinetuneResult:
    checkpoint_path: str
    final_loss: float
    sparsity: dict
    epochs: int


class FinetuneUseCase:
    """Fine-tunes a sparse model while preserving the mask topology."""

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
        output_path: str,
        epochs: int = 5,
        batches_per_epoch: int = 200,
        learning_rate: float = 1e-4,
    ) -> FinetuneResult:
        masks = self.mask_persister.load(mask_path)
        apply_masks_to_weights(masks, self.model)

        optimizer = torch.optim.SGD(
            [p for p in self.model._model.parameters() if p.requires_grad],
            lr=learning_rate,
            momentum=0.9,
        )

        best_loss = float("inf")
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in self.dataset.get_batches(batches_per_epoch):
                self.model.zero_grad()
                loss = self.model.forward_backward(batch)
                optimizer.step()
                apply_masks_to_weights(masks, self.model)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model._model.state_dict(), output_path)

        sparsity = count_sparsity(masks)

        return FinetuneResult(
            checkpoint_path=output_path,
            final_loss=best_loss,
            sparsity=sparsity,
            epochs=epochs,
        )
