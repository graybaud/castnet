"""Unit tests for logical rules R1-R6."""

import torch
import pytest
from domain.rules.logical_rules import (
    rule_r1_specialized,
    rule_r3_unique,
    rule_r4_impact,
    rule_r6_dynamic,
    apply_rules,
    unify_rules,
    rule_statistics,
)


class TestRuleR1:
    """R1: Specialized AND task-sensitive."""

    def test_both_high_kept(self):
        direction = torch.tensor([5.0, 3.0])
        grad_norm = torch.tensor([0.01, 0.0001])
        mask = rule_r1_specialized(direction, grad_norm)
        assert mask[0].item() is True   # both high
        assert mask[1].item() is False  # grad_norm too low

    def test_both_low_pruned(self):
        direction = torch.tensor([2.0])
        grad_norm = torch.tensor([0.0001])
        mask = rule_r1_specialized(direction, grad_norm)
        assert mask[0].item() is False

    def test_threshold_boundaries(self):
        # Juste en dessous et juste au dessus du seuil
        direction = torch.tensor([4.0, 4.1])
        grad_norm = torch.tensor([0.0011, 0.0011])  # > 0.001
        mask = rule_r1_specialized(direction, grad_norm)
        assert mask[0].item() is False  # direction=4.0 is NOT > 4.0
        assert mask[1].item() is True   # direction=4.1 IS > 4.0 AND grad_norm > 0.001


class TestRuleR3:
    """R3: Unique (low correlation)."""

    def test_low_correlation_kept(self):
        correlation = torch.tensor([0.1, 0.5, 0.39, 0.4])
        mask = rule_r3_unique(correlation)
        assert mask[0].item() is True
        assert mask[1].item() is False
        assert mask[2].item() is True   # 0.39 < 0.4
        assert mask[3].item() is False  # 0.4 is NOT < 0.4

    def test_negative_correlation(self):
        correlation = torch.tensor([-0.5])
        mask = rule_r3_unique(correlation)
        assert mask[0].item() is True


class TestRuleR4:
    """R4: High geometric impact."""

    def test_high_distortion_kept(self):
        distortion = torch.tensor([0.01, 0.0001, 0.001])
        mask = rule_r4_impact(distortion)
        assert mask[0].item() is True
        assert mask[1].item() is False
        assert mask[2].item() is False  # 0.001 is NOT > 0.001

    def test_zero_distortion(self):
        distortion = torch.tensor([0.0])
        mask = rule_r4_impact(distortion)
        assert mask[0].item() is False


class TestRuleR6:
    """R6: Strong temporal dynamics."""

    def test_mid_sequence_kept(self):
        latency = torch.tensor([0.5, 0.2, 0.3, 0.301])
        mask = rule_r6_dynamic(latency)
        assert mask[0].item() is True
        assert mask[1].item() is False
        assert mask[2].item() is False  # 0.3 is NOT > 0.3
        assert mask[3].item() is True


class TestApplyRules:

    def test_returns_all_four_rules(self):
        n = 10
        direction = torch.rand(n) * 10
        grad_norm = torch.rand(n)
        correlation = torch.rand(n)
        distortion = torch.rand(n)
        latency = torch.rand(n)

        result = apply_rules(direction, grad_norm, correlation, distortion, latency)

        assert set(result.keys()) == {"R1", "R3", "R4", "R6"}
        for mask in result.values():
            assert mask.shape == (n,)
            assert mask.dtype == torch.bool


class TestUnifyRules:

    def test_or_logic(self):
        masks = {
            "R1": torch.tensor([True, False, False]),
            "R3": torch.tensor([False, True, False]),
            "R4": torch.tensor([False, False, False]),
            "R6": torch.tensor([False, False, False]),
        }
        result = unify_rules(masks)
        assert result[0].item() == 1.0  # R1
        assert result[1].item() == 1.0  # R3
        assert result[2].item() == 0.0  # aucune regle

    def test_no_rules_match(self):
        masks = {
            "R1": torch.tensor([False]),
            "R3": torch.tensor([False]),
            "R4": torch.tensor([False]),
            "R6": torch.tensor([False]),
        }
        result = unify_rules(masks)
        assert result[0].item() == 0.0

    def test_all_rules_match(self):
        masks = {
            "R1": torch.tensor([True]),
            "R3": torch.tensor([True]),
            "R4": torch.tensor([True]),
            "R6": torch.tensor([True]),
        }
        result = unify_rules(masks)
        assert result[0].item() == 1.0

    def test_custom_weights(self):
        masks = {
            "R1": torch.tensor([True]),
            "R3": torch.tensor([False]),
        }
        weights = {"R1": 0.0, "R3": 1.0}
        result = unify_rules(masks, rule_weights=weights)
        assert result[0].item() == 0.0  # R1 weight = 0


class TestRuleStatistics:

    def test_counts(self):
        masks = {
            "R1": torch.tensor([True, False, True]),
            "R3": torch.tensor([False, True, False]),
        }
        stats = rule_statistics(masks, total_neurons=3)
        assert stats["R1"]["kept"] == 2
        assert stats["R1"]["ratio"] == pytest.approx(2/3, abs=0.001)
        assert stats["R3"]["kept"] == 1
