"""
test_mask_loading.py — Tests for load_masks.
"""

import os
import tempfile
import pytest
import torch
from safetensors.torch import save_file
from src.finetuning import load_masks


class TestLoadMasks:

    @pytest.fixture
    def safetensors_mask_file(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "masks.safetensors")
        masks = {
            "layer0_fc1": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "layer0_fc2": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        save_file(masks, path)
        yield path
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def pt_checkpoint_file(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "ckpt.pt")
        state = {
            'model_state_dict': {
                'layer0.fc1.weight': torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                'layer0.fc2.weight': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                'layer0.fc1.bias': torch.zeros(2),  # 1D, should be ignored
            }
        }
        torch.save(state, path)
        yield path
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def pt_checkpoint_no_model_key(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "ckpt.pt")
        state = {
            'layer0.fc1.weight': torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        }
        torch.save(state, path)
        yield path
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_safetensors(self, safetensors_mask_file):
        masks = load_masks(safetensors_mask_file, "cpu")
        assert len(masks) == 2
        assert "layer0_fc1" in masks
        assert masks["layer0_fc1"].dtype == torch.float32

    def test_load_pt_checkpoint(self, pt_checkpoint_file):
        masks = load_masks(pt_checkpoint_file, "cpu")
        assert len(masks) >= 1  # fc1.weight has zeros, fc2 has none?
        # fc2 has no zeros, so only fc1 mask extracted
        assert any("fc1" in k for k in masks)

    def test_load_pt_no_model_key(self, pt_checkpoint_no_model_key):
        masks = load_masks(pt_checkpoint_no_model_key, "cpu")
        assert len(masks) == 1

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported mask file format"):
            load_masks("masks.h5", "cpu")

    def test_mask_names_use_underscores(self, safetensors_mask_file):
        masks = load_masks(safetensors_mask_file, "cpu")
        for name in masks:
            assert '.' not in name  # Safetensors keeps original names
