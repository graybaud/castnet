"""Orchestration — Model evaluation entry point.

Modes:
  perplexity  — Evaluate perplexity and sparsity for a single mask
  sweep       — Evaluate perplexity at multiple keep fractions
  dense       — Evaluate dense baseline perplexity
  compare     — Compare dense vs sparse
"""

import hydra
from omegaconf import DictConfig
import torch

from infrastructure.models.huggingface import HuggingFaceWeightProvider
from infrastructure.data.providers import WikiTextProvider
from infrastructure.persistence.safetensors_persister import (
    SafetensorsScorePersister,
    SafetensorsMaskPersister,
)
from application.evaluate import EvaluateUseCase
from application.sweep import SparsitySweepUseCase


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {cfg.model} | Mode: {cfg.mode}")

    model = HuggingFaceWeightProvider(cfg.model, device)
    dataset = WikiTextProvider(cfg.model, cfg.max_len)
    mask_persister = SafetensorsMaskPersister()

    if cfg.mode == "dense":
        _run_dense(model, dataset, cfg)
    elif cfg.mode == "perplexity":
        _run_perplexity(model, dataset, mask_persister, cfg)
    elif cfg.mode == "sweep":
        _run_sweep(model, dataset, mask_persister, cfg)
    elif cfg.mode == "compare":
        _run_compare(model, dataset, mask_persister, cfg)


def _run_dense(model, dataset, cfg):
    print("Dense baseline evaluation...")
    model.model.eval()
    total_loss, total_tok = 0.0, 0
    for i, batch in enumerate(dataset.get_batches(cfg.num_batches)):
        batch = batch.to(model.model.device)
        with torch.no_grad():
            out = model.model(batch, labels=batch)
            loss = out.loss.item()
        total_loss += loss * batch.numel()
        total_tok += batch.numel()
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{cfg.num_batches}")

    from domain.metrics.perplexity import compute_perplexity_from_total
    perp = compute_perplexity_from_total(total_loss, total_tok)

    print(f"\n{'=' * 60}")
    print(f"  DENSE BASELINE")
    print(f"  Perplexity : {perp:.1f}")
    print(f"{'=' * 60}")


def _run_perplexity(model, dataset, mask_persister, cfg):
    use_case = EvaluateUseCase(model, dataset, mask_persister)
    result = use_case.execute(cfg.mask_path, num_batches=cfg.num_batches)

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Perplexity : {result.perplexity:.1f}")
    print(f"  Sparsity   : {result.sparsity['overall_sparsity_pct']}%")
    print(f"  Active     : {result.sparsity['total_active']:,} / {result.sparsity['total_connections']:,}")
    print(f"{'=' * 60}")


def _run_sweep(model, dataset, mask_persister, cfg):
    score_persister = SafetensorsScorePersister()
    use_case = SparsitySweepUseCase(model, dataset, score_persister, mask_persister)
    result = use_case.execute(cfg.scores_path, eval_batches=cfg.num_batches)

    print(f"\n{'=' * 60}")
    print(f"  SWEEP COMPLETE")
    print(f"  {'keep':<10} {'sparsity':<12} {'perplexity'}")
    print(f"  {'-' * 34}")
    for r in sorted(result.results, key=lambda x: x["perplexity"]):
        print(f"  {r['keep_fraction']:<10.2f} {r['sparsity_pct']:<12.1f} {r['perplexity']:<12.1f}")
    print(f"{'=' * 60}")


def _run_compare(model, dataset, mask_persister, cfg):
    print("[1/2] Dense baseline...")
    total_loss, total_tok = 0.0, 0
    model.model.eval()
    for i, batch in enumerate(dataset.get_batches(cfg.num_batches)):
        batch = batch.to(model.model.device)
        with torch.no_grad():
            out = model.model(batch, labels=batch)
            loss = out.loss.item()
        total_loss += loss * batch.numel()
        total_tok += batch.numel()
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{cfg.num_batches}")

    from domain.metrics.perplexity import compute_perplexity_from_total
    perp_dense = compute_perplexity_from_total(total_loss, total_tok)

    print("[2/2] Sparse evaluation...")
    use_case = EvaluateUseCase(model, dataset, mask_persister)
    result = use_case.execute(cfg.mask_path, num_batches=cfg.num_batches)

    ratio = result.perplexity / perp_dense if perp_dense > 0 else float("inf")
    if ratio < 1.5:
        verdict = "OK (<1.5x)"
    elif ratio < 3.0:
        verdict = "Acceptable (<3x)"
    elif ratio < 5.0:
        verdict = "Degraded (<5x)"
    else:
        verdict = "Broken"

    print(f"\n{'=' * 60}")
    print(f"  DENSE vs SPARSE COMPARISON")
    print(f"  Perplexity dense : {perp_dense:.1f}")
    print(f"  Perplexity sparse: {result.perplexity:.1f}")
    print(f"  Ratio           : {ratio:.1f}x {verdict}")
    print(f"  Sparsity        : {result.sparsity['overall_sparsity_pct']}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
