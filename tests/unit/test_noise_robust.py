"""Unit tests for noise robustness metric."""

from domain.metrics.noise_robust import compute_noise_degradation


class TestComputeNoiseDegradation:
    def test_basic(self):
        noisy = {"0.01": 35.0, "0.05": 40.0, "0.10": 50.0}
        result = compute_noise_degradation(30.0, noisy)
        assert result["clean_perplexity"] == 30.0
        assert result["noise_levels"]["0.01"]["perplexity_increase"] == 5.0
        assert result["noise_levels"]["0.10"]["perplexity_increase"] == 20.0

    def test_degradation_increases_with_noise(self):
        noisy = {"0.01": 32.0, "0.10": 60.0}
        result = compute_noise_degradation(30.0, noisy)
        d1 = result["noise_levels"]["0.01"]["degradation_ratio"]
        d2 = result["noise_levels"]["0.10"]["degradation_ratio"]
        assert d2 > d1

    def test_no_degradation(self):
        noisy = {"0.01": 30.0}
        result = compute_noise_degradation(30.0, noisy)
        assert result["noise_levels"]["0.01"]["perplexity_increase"] == 0.0

    def test_empty_noise_dict(self):
        result = compute_noise_degradation(30.0, {})
        assert result["clean_perplexity"] == 30.0
        assert result["noise_levels"] == {}

    def test_clean_perplexity_zero(self):
        noisy = {"0.01": 5.0}
        result = compute_noise_degradation(0.0, noisy)
        assert result["noise_levels"]["0.01"]["degradation_ratio"] > 0
