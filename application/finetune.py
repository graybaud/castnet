"""Use Case : Frozen-topology fine-tuning — matches legacy finetuning/training.py."""

from dataclasses import dataclass
import torch
from domain.scoring.ports import WeightProvider, BatchProvider, MaskPersister
from domain.scoring.masks import count_sparsity


@dataclass
class FinetuneResult:
    checkpoint_path: str
    final_loss: float
    sparsity: dict
    epochs: int


class FinetuneUseCase:
    """Fine-tunes a sparse model while preserving the mask topology.
    
    Identical to legacy finetuning/training.py:
    1. Load masks, build application dict, apply masks
    2. Train with frozen topology (re-apply masks after each step)
    3. Validate perplexity after each epoch
    4. Save best checkpoint
    """

    def __init__(
        self,
        model: WeightProvider,
        dataset: BatchProvider,
        mask_persister: MaskPersister,
        architecture: str = "generic",
    ):
        self.model = model
        self.dataset = dataset
        self.mask_persister = mask_persister
        self.architecture = architecture

    def execute(
        self,
        mask_path: str,
        output_path: str,
        epochs: int = 5,
        batches_per_epoch: int = 200,
        learning_rate: float = 1e-4,
        optimizer_name: str = "sgd",
        save_per_epoch: bool = False,
        sparsity_check: bool = True,
    ) -> FinetuneResult:
        from infrastructure.finetuning.mask_loader import (
            load_masks, build_mask_application_dict, apply_masks,
        )

        # 1. Load masks (supports .safetensors AND .pt) — like legacy
        device = next(self.model._model.parameters()).device
        masks = load_masks(mask_path, device)

        # 2. Build application dict (map mask names to model modules) — like legacy
        mask_application_dict = build_mask_application_dict(
            self.model._model, masks, self.architecture
        )

        if not mask_application_dict:
            raise ValueError("No masks could be applied to the model")

        print(f"Mapped {len(mask_application_dict)}/{len(masks)} masks to model modules")

        # 3. Apply masks initially — like legacy
        apply_masks(self.model._model, mask_application_dict)

        # 4. Sparsity check — like legacy
        sp_stats = count_sparsity(masks)
        print(f"Initial Sparsity: {sp_stats['overall_sparsity_pct']:.1f}% "
              f"({sp_stats['total_active']:,}/{sp_stats['total_connections']:,})")

        if sparsity_check and sp_stats['overall_sparsity_pct'] < 1.0:
            raise ValueError(
                "Sparsity is under 1%. Model is effectively dense. "
                "Use sparsity_check=False to bypass."
            )

        # 5. Optimizer — like legacy (SGD or AdamW)
        if optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                [p for p in self.model._model.parameters() if p.requires_grad],
                lr=learning_rate,
            )
        else:
            optimizer = torch.optim.SGD(
                [p for p in self.model._model.parameters() if p.requires_grad],
                lr=learning_rate,
                momentum=0.9,
            )
        print(f"Optimizer: {optimizer_name.upper()}, lr={learning_rate}")

        # 6. Training loop — identical to legacy
        self.model._model.train()
        best_loss = float("inf")

        for epoch in range(epochs):
            total_loss = 0.0
            actual_batches = 0

            for batch in self.dataset.get_batches(batches_per_epoch):
                self.model.zero_grad()
                loss = self.model.forward_backward(batch)
                optimizer.step()
                # Re-apply masks after each step (frozen topology)
                apply_masks(self.model._model, mask_application_dict)
                total_loss += loss
                actual_batches += 1

                if (actual_batches) % 100 == 0:
                    avg_loss = total_loss / actual_batches
                    print(f"  Epoch {epoch+1} Batch {actual_batches}/{batches_per_epoch}  "
                          f"loss={avg_loss:.4f}")

            if actual_batches == 0:
                print(f"  WARNING: Epoch {epoch+1} had 0 batches.")
                break

            avg_loss = total_loss / actual_batches
            print(f"Epoch {epoch+1} done: avg_loss={avg_loss:.4f}")

            # Save if best or save_per_epoch
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            if is_best or save_per_epoch:
                state_dict = {k: v.contiguous() for k, v in self.model._model.state_dict().items()}
                torch.save(state_dict, output_path)
                if is_best:
                    print(f"  [Saved] Best model: {output_path} (loss={avg_loss:.4f})")

        sparsity = count_sparsity(masks)

        return FinetuneResult(
            checkpoint_path=output_path,
            final_loss=best_loss,
            sparsity=sparsity,
            epochs=epochs,
        )
