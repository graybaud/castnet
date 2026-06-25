"""Tests for castnet.extraction.scorers.unified — Rule-Based Scoring"""
import pytest
import torch
import torch.nn as nn
from src.extraction.scorers.unified import (
    neuron_direction, neuron_correlation, neuron_distortion,
    neuron_latency, apply_logical_rules,
)

@pytest.fixture
def sample_W():
    W = torch.zeros(8, 4)
    W[0] = torch.tensor([3.0, 0.5, 0.2, 0.1])
    W[1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    W[4] = torch.tensor([5.0, 0.1, 0.1, 0.1])
    W[5] = torch.tensor([-2.0, -3.0, -1.0, -0.5])
    W[7] = torch.tensor([10.0, 0.01, 0.01, 0.01])
    return W

@pytest.fixture
def sample_rule_scores():
    return {"fc1": {
        'M2': torch.tensor([6.3, 1.0, 2.5, 1.0, 12.5, 4.0, 1.2, 50.0]),
        'M4': torch.tensor([0.8, 0.1, 0.5, 0.05, 0.9, 0.3, 0.01, 0.95]),
        'M9': torch.tensor([0.3, 0.7, 0.5, 0.9, 0.2, 0.6, 0.8, 0.1]),
        'M10': torch.tensor([0.05, 0.001, 0.02, 0.0, 0.1, 0.01, 0.0, 0.08]),
        'latency': torch.tensor([0.5, 0.2, 0.8, 0.1, 0.9, 0.4, 0.0, 0.95]),
    }}

class TestNeuronDirection:
    def test_strong_direction(self, sample_W):
        d = neuron_direction(sample_W[4])
        assert 3.5 < d < 4.0
    def test_weak_direction(self, sample_W):
        d = neuron_direction(sample_W[1])
        assert 0.99 < d < 1.01
    def test_negative_weights(self, sample_W):
        d = neuron_direction(sample_W[5])
        assert d > 1.5
    def test_all_positive(self, sample_W):
        for i in [0,1,4,5,7]:
            assert neuron_direction(sample_W[i]) >= 1.0

class TestNeuronCorrelation:
    def test_self_excluded(self, sample_W):
        c = neuron_correlation(sample_W, 0)
        assert c < 1.0
    def test_with_identical_weights(self):
        W = torch.ones(8, 4)
        c = neuron_correlation(W, 0)
        assert c > 0.85

class TestNeuronDistortion:
    def test_positive(self, sample_W):
        x_in = torch.randn(50, 4)
        x_out = torch.randn(50, 8)
        for i in [0,1,4,5,7]:
            assert neuron_distortion(sample_W, 0.0, x_in, x_out, i) >= 0.0

class TestNeuronLatency:
    def test_middle_activation(self):
        acts = torch.zeros(50, 4)
        acts[25, 0] = 10.0
        assert neuron_latency(acts)[0] > 0.9
    def test_edge_activation(self):
        acts = torch.zeros(50, 4)
        acts[0, 0] = 10.0
        assert neuron_latency(acts)[0] < 0.1

class TestApplyLogicalRules:
    def test_returns_dict(self, sample_rule_scores):
        masks = apply_logical_rules(sample_rule_scores, 0.30)
        assert "fc1" in masks
    def test_mask_is_binary(self, sample_rule_scores):
        m = apply_logical_rules(sample_rule_scores, 0.30)["fc1"]
        assert torch.all((m == 0) | (m == 1))
    def test_keep_100_pct(self, sample_rule_scores):
        masks = apply_logical_rules(sample_rule_scores, 1.0)
        assert masks["fc1"].sum().item() == 8
    def test_monotonic(self, sample_rule_scores):
        k30 = apply_logical_rules(sample_rule_scores, 0.30)["fc1"].sum().item()
        k50 = apply_logical_rules(sample_rule_scores, 0.50)["fc1"].sum().item()
        assert k50 >= k30
    def test_some_kept_some_pruned(self, sample_rule_scores):
        m = apply_logical_rules(sample_rule_scores, 0.30)["fc1"]
        assert 0 < m.sum() < 8

class TestFunctionalSweep:
    @pytest.fixture
    def mock_model_and_data(self):
        class RealisticModel:
            def __init__(self):
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 8)
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
            def train(self): pass
            def eval(self): pass
            def zero_grad(self): pass
            def __call__(self, x, labels=None):
                loss = nn.functional.mse_loss(self.fc2(torch.relu(self.fc1(x))), torch.zeros(x.shape[0], 8))
                return type("obj", (object,), {"loss": loss})()
        model = RealisticModel()
        ffn = [("fc1", model.fc1), ("fc2", model.fc2)]
        dataset = [{"input_ids": torch.randn(2, 16)} for _ in range(30)]
        return model, ffn, dataset

    def test_sweep_keep_fractions(self, mock_model_and_data):
        from src.extraction.scorers.unified import accumulate_unified_scores
        model, ffn, dataset = mock_model_and_data
        results = {}
        for kf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = accumulate_unified_scores(model, ffn, dataset, 5, 'cpu', 20, 4, kf)
            total_active = sum(m.sum().item() for m in result.values())
            total_conn = sum(m.numel() for m in result.values())
            results[kf] = (1 - total_active / total_conn) * 100
        sparsities = [results[k] for k in sorted(results)]
        for i in range(len(sparsities)-1):
            assert sparsities[i] >= sparsities[i+1] - 1

    def test_per_layer_consistency(self, mock_model_and_data):
        from src.extraction.scorers.unified import accumulate_unified_scores
        model, ffn, dataset = mock_model_and_data
        result = accumulate_unified_scores(model, ffn, dataset, 5, 'cpu', 20, 4, 0.3)
        for name, mask in result.items():
            assert (mask.sum(dim=1) > 0).sum().item() > 0
