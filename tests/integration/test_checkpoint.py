"""
test_checkpoint.py — Tests for load_checkpoint.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint from .pt or .safetensors."""
    import torch
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    elif checkpoint_path.endswith('.pt'):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=False)
    return model


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc1(x)


class TestLoadCheckpoint:

    @pytest.fixture
    def safetensors_ckpt(self):
        from safetensors.torch import save_file
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "ckpt.safetensors")
        state = {"fc1.weight": torch.randn(8, 4), "fc1.bias": torch.randn(8)}
        save_file(state, path)
        yield path
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def pt_ckpt(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "ckpt.pt")
        state = {"fc1.weight": torch.randn(8, 4), "fc1.bias": torch.randn(8)}
        torch.save(state, path)
        yield path
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_safetensors(self, safetensors_ckpt):
        model = DummyModel()
        original = model.fc1.weight.clone()
        model = load_checkpoint(model, safetensors_ckpt, "cpu")
        # Weights should have changed
        assert not torch.equal(model.fc1.weight, original)

    def test_load_pt(self, pt_ckpt):
        model = DummyModel()
        original = model.fc1.weight.clone()
        model = load_checkpoint(model, pt_ckpt, "cpu")
        assert not torch.equal(model.fc1.weight, original)

    def test_strict_false_allows_missing(self, safetensors_ckpt):
        """Missing keys should not raise error (strict=False)."""
        model = nn.Linear(10, 10)  # No matching keys at all
        model = load_checkpoint(model, safetensors_ckpt, "cpu")
        # Should not raise, just warn

    def test_unsupported_format(self):
        model = DummyModel()
        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            load_checkpoint(model, "ckpt.h5", "cpu")
