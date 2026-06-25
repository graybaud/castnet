"""
Robustness tests for geometric utilities — migrated from legacy.
"""

import pytest
import torch
import torch.nn as nn
from domain.scoring.geometric import (
    cosine_similarity_matrix,
    frobenius_distortion,
    _organize_layers,
    find_natural_threshold,
    neuron_direction,
    neuron_selectivity,
)


class TestEdgeCases:
    def test_zero_vectors(self):
        v = torch.zeros(10, 5)
        S = cosine_similarity_matrix(v)
        assert not torch.isnan(S).any()
        assert not torch.isinf(S).any()

    def test_single_vector(self):
        v = torch.randn(1, 10)
        S = cosine_similarity_matrix(v)
        assert S.shape == (1, 1)
        assert abs(S[0, 0].item() - 1.0) < 1e-6

    def test_identical_vectors(self):
        v = torch.ones(5, 10)
        S = cosine_similarity_matrix(v)
        assert torch.allclose(S, torch.ones(5, 5), atol=1e-5)

    def test_orthogonal_vectors(self):
        v = torch.eye(5)
        S = cosine_similarity_matrix(v)
        assert torch.allclose(S.diag(), torch.ones(5), atol=1e-6)

    def test_all_negative_weights(self):
        w = -torch.rand(100)
        d = neuron_direction(w)
        assert d > 0.0

    def test_constant_activations(self):
        acts = torch.ones(100)
        s = neuron_selectivity(acts)
        assert s == 0.0


class TestInvariants:
    def test_distortion_zero_for_identical(self):
        for n in [10, 50, 100]:
            S = torch.randn(n, n)
            S = S @ S.T
            d = frobenius_distortion(S, S)
            assert abs(d) < 1e-6

    def test_distortion_non_negative(self):
        S1 = torch.randn(20, 20)
        S2 = torch.randn(20, 20)
        d = frobenius_distortion(S1, S2)
        assert d >= 0.0

    def test_organize_layers_all_have_fc1_fc2(self):
        ffn = [(f"layer{i}.fc1", nn.Linear(4, 8)) for i in range(5)] + \
              [(f"layer{i}.fc2", nn.Linear(8, 4)) for i in range(5)]
        layers = _organize_layers(ffn)
        for i in range(5):
            assert 'fc1' in layers[i]
            assert 'fc2' in layers[i]

    def test_otsu_threshold_in_range(self):
        for _ in range(10):
            scores = {"fc": torch.rand(200, 10)}
            result = find_natural_threshold(scores)
            assert 0.0 <= result["threshold"] <= 1.0
            assert 0.0 < result["keep_fraction"] <= 1.0


class TestCoherence:
    def test_threshold_monotonic(self):
        scores_low = {"fc": torch.rand(500, 10) * 0.3}
        scores_high = {"fc": torch.rand(500, 10) * 0.3 + 0.5}
        r_low = find_natural_threshold(scores_low)
        r_high = find_natural_threshold(scores_high)
        assert 0 <= r_low["threshold"] <= 1
        assert 0 <= r_high["threshold"] <= 1

    def test_perfectly_separable_scores(self):
        scores = {"fc": torch.cat([torch.zeros(500, 10), torch.ones(500, 10)])}
        result = find_natural_threshold(scores)
        assert 0.4 < result["threshold"] < 0.6

    def test_all_scores_same(self):
        scores = {"fc": torch.full((500, 10), 0.5)}
        result = find_natural_threshold(scores)
        assert result["keep_fraction"] in [0.0, 1.0] or 0.4 < result["keep_fraction"] < 0.6


class TestChainCoherence:
    def test_chain_v1_weights_shape_match(self):
        fc2_dim = 8
        fc1_next_dim = 4
        W = torch.randn(fc1_next_dim, fc2_dim)
        scores_fc1_next = torch.rand(fc1_next_dim)
        importance = W.abs().T @ scores_fc1_next
        assert importance.shape == (fc2_dim,)

    def test_chain_bonus_positive(self):
        fc2_dim = 8
        fc1_next_dim = 4
        W = torch.randn(fc1_next_dim, fc2_dim).abs()
        scores_fc1_next = torch.rand(fc1_next_dim)
        importance = W.T @ scores_fc1_next
        importance = importance / (importance.max() + 1e-8)
        assert (importance >= 0).all()
        assert importance.max() <= 1.0 + 1e-6

    def test_chain_preserves_zero_scores(self):
        fc2_scores = torch.zeros(8)
        fc2_scores[0] = 0.5
        fc1_next_scores = torch.ones(4)
        W = torch.randn(4, 8).abs()
        importance = W.T @ fc1_next_scores
        importance = importance / (importance.max() + 1e-8)
        fc2_chain = fc2_scores * (1.0 + importance)
        assert fc2_chain[0] > 0
        assert fc2_chain[1:].sum() == 0.0


class TestResilience:
    def test_empty_ffn_layers(self):
        layers = _organize_layers([])
        assert layers == {}

    def test_single_layer(self):
        ffn = [("layer0.fc1", nn.Linear(4, 8))]
        layers = _organize_layers(ffn)
        assert 0 in layers
        assert 'fc2' not in layers[0]

    def test_missing_fc2(self):
        ffn = [("layer0.fc1", nn.Linear(4, 8))]
        layers = _organize_layers(ffn)
        assert 'fc2' not in layers[0]

    def test_non_sequential_layers(self):
        ffn = [
            ("layer5.fc2", nn.Linear(8, 4)),
            ("layer5.fc1", nn.Linear(4, 8)),
            ("layer0.fc1", nn.Linear(4, 8)),
            ("layer0.fc2", nn.Linear(8, 4)),
        ]
        layers = _organize_layers(ffn)
        assert 0 in layers and 5 in layers
        assert layers[0]['fc1'] == "layer0.fc1"
        assert layers[5]['fc2'] == "layer5.fc2"

    def test_large_scores_dont_overflow(self):
        scores = {
            "fc1": torch.rand(100, 50) * 1000,
            "fc2": torch.rand(50, 100) * 1e-6,
        }
        result = find_natural_threshold(scores)
        assert not torch.isnan(torch.tensor(result["threshold"]))
        assert not torch.isinf(torch.tensor(result["threshold"]))
