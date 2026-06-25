"""Fixtures for legacy tests."""

import pytest
import torch
import torch.nn as nn


class FFNModel(nn.Module):
    """Mock model with fc1 (4->8), fc2 (8->4), and an attention layer (ignored)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.attn = nn.Linear(4, 4)  # should be ignored

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class LlamaModel(nn.Module):
    """Mock LLaMA-style model with gate_proj, up_proj, down_proj."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32)
        self.up_proj = nn.Linear(16, 32)
        self.down_proj = nn.Linear(32, 16)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


@pytest.fixture
def ffn_model():
    return FFNModel()


@pytest.fixture
def llama_model():
    return LlamaModel()
