"""Property-based tests — Mathematical invariants that must always hold."""

import torch
import pytest
from domain.scoring.strategies import (
    MagnitudeStrategy,
    GradientStrategy,
    WandaStrategy,
    GPSStrategy,
    get_strategy,
)
from domain.scoring.masks import apply_percentile_mask, count_sparsity


class FakeWeightProvider:
    def __init__(self, W, bias=None, grad=None):
        self._W, self._bias, self._grad = W, bias, grad
    def get_weight(self, name): return self._W
    def get_bias(self, name): return self._bias
    def get_gradient(self, name): return self._grad
    def layer_names(self): return ["test"]
    def forward_backward(self, batch): return 1.0
    def zero_grad(self): pass


class FakeActivationProvider:
    def __init__(self, X_in, X_out=None):
        self._X_in = X_in
        self._X_out = X_out if X_out is not None else X_in
    def get_input_activations(self, name): return self._X_in
    def get_output_activations(self, name): return self._X_out
    def attach(self): pass
    def detach(self): pass


# ---------------------------------------------------------------------------
# Proprietes de normalisation
# ---------------------------------------------------------------------------

class TestNormalizationInvariants:

    @pytest.mark.parametrize("strategy_name", ["magnitude", "gradient", "wanda", "gps"])
    def test_scores_in_01(self, strategy_name):
        """Tous les scores doivent etre dans [0, 1]."""
        W = torch.randn(10, 20)
        X_in = torch.randn(50, 20)
        X_out = torch.randn(50, 10)
        grad = torch.randn(10, 20)

        strategy = get_strategy(strategy_name)
        scores = strategy.calculate(
            FakeWeightProvider(W, grad=grad),
            FakeActivationProvider(X_in, X_out),
            "test",
        )

        assert scores.min() >= 0.0, f"{strategy_name}: min < 0"
        assert scores.max() <= 1.0 + 1e-6, f"{strategy_name}: max > 1"

    @pytest.mark.parametrize("strategy_name", ["magnitude", "gradient", "wanda", "gps"])
    def test_max_is_exactly_one(self, strategy_name):
        """Apres normalisation, le max doit etre exactement 1.0."""
        W = torch.randn(10, 20).abs() + 0.1  # eviter tous zeros
        X_in = torch.randn(50, 20).abs() + 0.1
        X_out = torch.randn(50, 10).abs() + 0.1
        grad = torch.randn(10, 20).abs() + 0.1

        strategy = get_strategy(strategy_name)
        scores = strategy.calculate(
            FakeWeightProvider(W, grad=grad),
            FakeActivationProvider(X_in, X_out),
            "test",
        )

        assert torch.allclose(scores.max(), torch.tensor(1.0)), \
            f"{strategy_name}: max = {scores.max()}"


# ---------------------------------------------------------------------------
# Proprietes de symetrie
# ---------------------------------------------------------------------------

class TestSymmetryInvariants:

    def test_magnitude_scale_invariant(self):
        """Magnitude(W) et Magnitude(k*W) doivent etre identiques."""
        W = torch.randn(5, 10)
        k = 3.0

        strategy = MagnitudeStrategy()
        s1 = strategy.calculate(FakeWeightProvider(W), FakeActivationProvider(None), "test")
        s2 = strategy.calculate(FakeWeightProvider(W * k), FakeActivationProvider(None), "test")

        assert torch.allclose(s1, s2, atol=1e-6)

    def test_magnitude_permutation_equivariant(self):
        """Permuter les lignes de W doit permuter les lignes du score."""
        W = torch.randn(4, 6)
        perm = torch.randperm(4)

        strategy = MagnitudeStrategy()
        s1 = strategy.calculate(FakeWeightProvider(W), FakeActivationProvider(None), "test")
        s2 = strategy.calculate(FakeWeightProvider(W[perm]), FakeActivationProvider(None), "test")

        assert torch.allclose(s1[perm], s2, atol=1e-6)


# ---------------------------------------------------------------------------
# Proprietes des masques
# ---------------------------------------------------------------------------

class TestMaskInvariants:

    def test_mask_binary(self):
        """Les masques doivent etre strictement binaires (0 ou 1)."""
        scores = torch.rand(10, 10)
        mask = apply_percentile_mask(scores, keep_fraction=0.3)
        unique = mask.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_idempotence(self):
        """Appliquer deux fois le meme masque ne change rien."""
        scores = torch.rand(10, 10)
        mask1 = apply_percentile_mask(scores, keep_fraction=0.4)
        mask2 = apply_percentile_mask(scores, keep_fraction=0.4)
        assert torch.equal(mask1, mask2)

    def test_monotonicity(self):
        """keep_fraction plus grand => plus de connexions gardees."""
        scores = torch.rand(10, 10)
        n_03 = int(apply_percentile_mask(scores, 0.3).sum())
        n_05 = int(apply_percentile_mask(scores, 0.5).sum())
        n_08 = int(apply_percentile_mask(scores, 0.8).sum())
        assert n_03 <= n_05 <= n_08

    def test_count_sparsity_consistency(self):
        """count_sparsity doit etre coherent avec le comptage manuel."""
        masks = {
            "L0": torch.eye(3),  # 3 actif / 9 total
        }
        stats = count_sparsity(masks)
        expected_sparsity = (1 - 3/9) * 100  # 66.666... arrondi a 66.7
        assert stats["overall_sparsity_pct"] == pytest.approx(expected_sparsity, abs=0.1)
        assert stats["total_active"] == 3
        assert stats["total_connections"] == 9
