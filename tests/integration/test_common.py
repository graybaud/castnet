"""Tests for common utilities — migrated to domain layer."""

import torch
import numpy as np
from domain.scoring.masks import apply_percentile_mask
from domain.analysis.correlation import pearson_correlation
from infrastructure.hooks.coactivation import CoActivationHook


def test_apply_percentile_mask():
    score = torch.tensor([[0.9, 0.1, 0.5], [0.3, 0.7, 0.2]])
    mask = apply_percentile_mask(score, 0.5)
    assert mask.sum().item() == 3
    flat = score.flatten()
    kept = flat[mask.flatten() > 0]
    pruned = flat[mask.flatten() == 0]
    assert kept.min() >= pruned.max()


def test_apply_percentile_mask_empty():
    score = torch.zeros(4, 4)
    mask = apply_percentile_mask(score, 0.5)
    assert mask.sum().item() == 0


def test_pearson_perfect():
    assert pearson_correlation([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_pearson_uncorrelated():
    assert pearson_correlation([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0


def test_coactivation_hook_init():
    W = torch.randn(64, 32)
    hook = CoActivationHook(W, "test_layer")
    assert hook.freq_matrix.shape == (64, 32)
    assert hook.total_tokens == 0


def test_coactivation_hook_finalize():
    W = torch.randn(4, 4)
    hook = CoActivationHook(W, "test")
    hook.freq_matrix = torch.tensor([[8.0, 0.0], [4.0, 4.0]])
    hook.total_tokens = 8
    hook.finalize()
    expected = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    assert torch.allclose(hook.freq_matrix, expected)
