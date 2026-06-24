"""Integration tests — Full pipeline with tiny model."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestFullPipeline:

    def test_pipeline_config_loads(self):
        """Verify the pipeline config can be loaded."""
        from hydra import compose, initialize_config_dir

        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs"
        )
        config_dir = os.path.abspath(config_dir)

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="pipeline")
            assert cfg.model is not None
            assert cfg.run_extract is True

    def test_pipeline_with_tiny_model(self):
        """Run extract -> masks -> evaluate with tiny model on CPU."""
        from hydra import compose, initialize_config_dir
        from infrastructure.models.huggingface import HuggingFaceWeightProvider
        from infrastructure.data.providers import WikiTextProvider
        from infrastructure.hooks.activation_collector import ActivationCollector
        from infrastructure.persistence.safetensors_persister import (
            SafetensorsScorePersister,
            SafetensorsMaskPersister,
        )
        from domain.scoring.strategies import get_strategy
        from application.extract_scores import ExtractScoresUseCase
        from application.generate_masks import GenerateMasksUseCase
        from application.evaluate import EvaluateUseCase
        from application.pipeline import CastNetPipeline

        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs"
        )
        config_dir = os.path.abspath(config_dir)

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="pipeline",
                overrides=[
                    "model=sshleifer/tiny-gpt2",
                    "device=cpu",
                    "extract.method=magnitude",
                    "extract.num_batches=2",
                    "extract.max_len=32",
                    "masks.keep_fraction=0.5",
                    "run_finetune=false",
                    "run_evaluate=true",
                ],
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            scores_path = os.path.join(tmpdir, "scores.safetensors")
            masks_path = os.path.join(tmpdir, "masks.safetensors")
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            # Infrastructure
            model = HuggingFaceWeightProvider(cfg.model, cfg.device)
            dataset = WikiTextProvider(cfg.model, cfg.extract.max_len)

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
            evaluate_uc = EvaluateUseCase(model, dataset, mask_persister)

            pipeline = CastNetPipeline(extract_uc, mask_uc, None, evaluate_uc)

            result = pipeline.run(
                scores_path=scores_path,
                masks_path=masks_path,
                checkpoint_path=checkpoint_path,
                num_batches=cfg.extract.num_batches,
                keep_fraction=cfg.masks.keep_fraction,
                eval_batches=2,
            )

            assert result.extract is not None
            assert result.mask is not None
            assert result.finetune is None
            assert result.evaluate is not None
            assert result.evaluate.perplexity > 0
            assert result.evaluate.sparsity["overall_sparsity_pct"] > 0
