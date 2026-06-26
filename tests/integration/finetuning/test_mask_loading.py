"""Tests for mask loading — migrated from legacy."""

import os
import tempfile
import pytest
import torch
from safetensors.torch import save_file
from infrastructure.finetuning.mask_loader import (
    load_masks,
    convert_mask_name_to_model,
    build_mask_application_dict,
    apply_masks,
)


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
                'layer0.fc1.bias': torch.zeros(2),
            }
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
        assert len(masks) >= 1

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported mask file format"):
            load_masks("masks.h5", "cpu")


class TestConvertMaskNameToModel:

    def test_phi2(self):
        result = convert_mask_name_to_model("layer0_fc1", "phi2")
        assert "model.layers.0.mlp.fc1" in result

    def test_opt(self):
        result = convert_mask_name_to_model("layer3_fc2", "opt")
        assert "model.decoder.layers.3.fc2" in result

    def test_llama_gate_proj(self):
        result = convert_mask_name_to_model("layer1_gate_proj", "llama")
        assert "model.layers.1.mlp.gate_proj" in result


class TestBuildMaskApplicationDict:

    def test_maps_modules(self):
        import torch.nn as nn
        
        # Build a model with a nested structure that matches phi2 naming
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.fc1 = nn.Linear(4, 8)
                self.mlp.fc2 = nn.Linear(8, 4)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([MockLayer()])
        
        model = MockModel()
        masks = {"layer0_fc1": torch.ones(8, 4)}
        result = build_mask_application_dict(model, masks, "phi2")
        assert isinstance(result, dict)
        # Should find the module
        assert len(result) == 1


class TestApplyMasks:

    def test_apply_masks(self):
        import torch.nn as nn
        linear = nn.Linear(2, 3)
        linear.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        
        mask_dict = {"test": (linear, mask)}
        apply_masks(linear, mask_dict)  # model is any nn.Module with the submodule
        
        expected = torch.tensor([[1.0, 0.0], [0.0, 4.0], [5.0, 0.0]])
        assert torch.equal(linear.weight.data, expected)
