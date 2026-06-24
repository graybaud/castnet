"""Integration tests — Hydra orchestration with real (tiny) model."""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omegaconf import OmegaConf


@pytest.fixture
def temp_output():
    """Create a temporary output file that gets cleaned up."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestHydraExtract:

    def test_extract_config_loads(self):
        """Verify the extract config can be loaded and merged."""
        from hydra import compose, initialize_config_dir

        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs"
        )
        config_dir = os.path.abspath(config_dir)

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="extract")
            assert cfg.model == "microsoft/phi-2"
            assert cfg.method in ("magnitude", "gradient", "wanda", "gps")
            assert cfg.num_batches > 0

    def test_extract_config_override(self):
        """Verify config can be overridden programmatically."""
        from hydra import compose, initialize_config_dir

        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs"
        )
        config_dir = os.path.abspath(config_dir)

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="extract",
                overrides=[
                    "model=sshleifer/tiny-gpt2",
                    "method=magnitude",
                    "num_batches=2",
                    "max_len=32",
                    "device=cpu",
                ],
            )
            assert cfg.model == "sshleifer/tiny-gpt2"
            assert cfg.method == "magnitude"
            assert cfg.num_batches == 2
            assert cfg.max_len == 32

    def test_full_extraction_pipeline(self, temp_output):
        """Run extract -> masks end-to-end with tiny model on CPU."""
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

        # Load config
        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs"
        )
        config_dir = os.path.abspath(config_dir)

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="extract",
                overrides=[
                    "model=sshleifer/tiny-gpt2",
                    "method=magnitude",
                    "num_batches=2",
                    "max_len=32",
                    "device=cpu",
                ],
            )

        # Step 1: Extract
        model = HuggingFaceWeightProvider(cfg.model, cfg.device)
        dataset = WikiTextProvider(cfg.model, cfg.max_len)

        real_names = list(dict(model._model.named_modules()).keys())
        ffn_names = [
            n for n in real_names
            if any(p in n.lower() for p in HuggingFaceWeightProvider.FFN_PATTERNS)
        ]
        collector = ActivationCollector(model._model, ffn_names)
        score_persister = SafetensorsScorePersister()

        strategy = get_strategy(cfg.method)
        extract_uc = ExtractScoresUseCase(
            strategy, model, dataset, collector, score_persister
        )
        result = extract_uc.execute(cfg.num_batches, temp_output)

        assert result.num_layers > 0
        assert result.num_batches == 2
        assert os.path.exists(temp_output)

        # Step 2: Generate masks
        mask_persister = SafetensorsMaskPersister()
        mask_uc = GenerateMasksUseCase(score_persister, mask_persister)

        masks_path = temp_output.replace(".safetensors", "_masks.safetensors")
        mask_result = mask_uc.execute(temp_output, masks_path, keep_fraction=0.5)

        assert mask_result.sparsity["total_active"] > 0
        assert os.path.exists(masks_path)

        # Cleanup
        if os.path.exists(masks_path):
            os.unlink(masks_path)
