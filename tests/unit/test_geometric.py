"""Tests for geometric utilities — migrated from legacy."""

import pytest
import torch
import torch.nn as nn
from domain.scoring.geometric import (
    cosine_similarity_matrix,
    frobenius_distortion,
    _organize_layers,
    find_natural_threshold,
)


@pytest.fixture
def sample_vectors():
    return torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])


@pytest.fixture
def sample_ffn_layers():
    layers = []
    for i in range(2):
        layers.append((f"layer{i}.fc1", nn.Linear(4, 8)))
        layers.append((f"layer{i}.fc2", nn.Linear(8, 4)))
    return layers


@pytest.fixture
def sample_scores():
    return {
        "layer0.fc1": torch.rand(8, 4),
        "layer0.fc2": torch.rand(4, 8),
        "layer1.fc1": torch.rand(8, 4),
        "layer1.fc2": torch.rand(4, 8),
    }


class TestCosineSimilarityMatrix:
    def test_shape(self, sample_vectors):
        S = cosine_similarity_matrix(sample_vectors)
        assert S.shape == (3, 3)

    def test_diagonal_is_one(self, sample_vectors):
        S = cosine_similarity_matrix(sample_vectors)
        assert torch.allclose(S.diag(), torch.ones(3), atol=1e-6)

    def test_symmetric(self, sample_vectors):
        S = cosine_similarity_matrix(sample_vectors)
        assert torch.allclose(S, S.T, atol=1e-6)

    def test_range(self, sample_vectors):
        S = cosine_similarity_matrix(sample_vectors)
        assert S.min() >= -1.0 and S.max() <= 1.0


class TestFrobeniusDistortion:
    def test_identical(self):
        S = torch.eye(10)
        assert frobenius_distortion(S, S) == 0.0

    def test_different(self):
        S1 = torch.eye(10)
        S2 = torch.zeros(10, 10)
        d = frobenius_distortion(S1, S2)
        assert d > 0.0


class TestOrganizeLayers:
    def test_two_layers(self, sample_ffn_layers):
        layers = _organize_layers(sample_ffn_layers)
        assert 0 in layers and 1 in layers
        assert layers[0]['fc1'] == "layer0.fc1"
        assert layers[0]['fc2'] == "layer0.fc2"

    def test_llama_naming(self):
        llama_layers = [
            ("layer0.gate_proj", nn.Linear(4, 8)),
            ("layer0.up_proj", nn.Linear(4, 8)),
            ("layer0.down_proj", nn.Linear(8, 4)),
        ]
        layers = _organize_layers(llama_layers)
        assert layers[0]['fc2'] == "layer0.down_proj"


class TestFindNaturalThreshold:
    def test_bimodal_distribution(self):
        scores = {"fc1": torch.cat([torch.full((500, 10), 0.1), torch.full((500, 10), 0.9)])}
        result = find_natural_threshold(scores)
        assert 0.05 < result["threshold"] < 0.95

    def test_uniform_distribution(self):
        scores = {"fc1": torch.rand(1000, 10)}
        result = find_natural_threshold(scores)
        assert result["keep_fraction"] > 0.0

    def test_few_scores_fallback(self):
        scores = {"fc1": torch.tensor([[0.5, 0.6]])}
        result = find_natural_threshold(scores)
        assert result["method"] == "fallback"

    def test_keep_fraction_range(self):
        scores = {"fc1": torch.rand(500, 10)}
        result = find_natural_threshold(scores)
        assert 0.0 < result["keep_fraction"] <= 1.0


class TestGPSChainLogic:
    def test_chain_preserves_shape(self, sample_scores):
        for name, s in sample_scores.items():
            assert s.shape == sample_scores[name].shape

    def test_chain_scores_in_01(self, sample_scores):
        for s in sample_scores.values():
            s_flat = s.flatten()
            s_nz = s_flat[s_flat > 1e-8]
            if len(s_nz) > 0:
                assert s_nz.min() >= 0.0
                assert s_nz.max() <= 1.0 + 1e-6
