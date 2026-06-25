"""Unit tests for neuron activation frequency."""

import torch
from domain.metrics.neuron_freq import (
    compute_activation_frequency,
    compute_activation_frequency_all_layers,
)


class TestComputeActivationFrequency:
    def test_all_active(self):
        acts = torch.ones(100, 10)
        result = compute_activation_frequency(acts)
        assert result["dead_neurons"] == 0
        assert result["mean_frequency"] == 1.0

    def test_all_dead(self):
        acts = torch.zeros(100, 10)
        result = compute_activation_frequency(acts, threshold=0.0)
        assert result["dead_neurons"] == 10

    def test_half_active(self):
        acts = torch.cat([torch.ones(100, 5), torch.zeros(100, 5)], dim=1)
        result = compute_activation_frequency(acts)
        assert result["dead_neurons"] == 5
        assert result["mean_frequency"] == 1.0

    def test_3d_input(self):
        acts = torch.ones(10, 20, 5)  # [B, S, d_out]
        result = compute_activation_frequency(acts)
        assert result["tokens_processed"] == 200

    def test_threshold(self):
        acts = torch.tensor([[0.05, 0.5], [0.02, 0.8]])
        result = compute_activation_frequency(acts, threshold=0.1)
        assert result["dead_neurons"] == 1  # first neuron never > 0.1


class TestComputeActivationFrequencyAllLayers:
    def test_multiple_layers(self):
        acts_dict = {
            "L0": torch.rand(50, 10),
            "L1": torch.zeros(50, 5),
        }
        result = compute_activation_frequency_all_layers(acts_dict)
        assert result["total_neurons"] == 15
        assert result["total_dead_neurons"] > 0
        assert "per_layer" in result

    def test_empty(self):
        result = compute_activation_frequency_all_layers({})
        assert result["total_neurons"] == 0
