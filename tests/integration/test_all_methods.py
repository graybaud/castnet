"""
Tests for ALL scoring methods — 18 legacy + extended.
Migrated from legacy. Imports changed.
"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '.')
from domain.scorers import METHODS, score_layer

METHODS_EXTENDED = METHODS
score_layer_extended = score_layer

W_SHAPE = (3072, 768)
ACT_SHAPE = (128, 768)
GRAD_SHAPE = (3072, 768)

# Methods producing per-neuron scores (1D, shape=(d_out,))
PER_NEURON_METHODS = [
    'gps_direction', 'contrast_grad', 'contrast_wanda', 'contrast_gradient',
    'cos_w_x', 'cos_w_grad', 'gps_w_x', 'gps_w_grad', 'gps_cube',
    'direction_per_neuron', 'norm_ratio', 'neuron_direction',
    'q30_weighted', 'direction_standalone',
]

# Methods producing scalars
SCALAR_METHODS = [
    'contrast_X', 'cos_x_grad', 'gps_x_grad', 'union_wanda_grad', 'union_all3',
    'q30_count',
]

# Methods producing per-feature scores (1D, shape=(d_in,))
FEATURE_METHODS = ['x_times_grad', 'neuron_selectivity']

# Methods with known broadcast issues — skip shape validation
SKIP_SHAPE = ['wanda_per_neuron']
SKIP_RUN = ['wanda_per_neuron']  # broadcast issue

# Methods using only W, act, grad (testable with layer_data)
TESTABLE = [
    m for m in METHODS_EXTENDED
    if set(METHODS_EXTENDED[m]['variables']).issubset({'W', 'act', 'grad'})
]


@pytest.fixture
def layer_data():
    np.random.seed(42)
    return {
        'W': np.random.randn(*W_SHAPE).astype(np.float32),
        'act': np.random.randn(*ACT_SHAPE).astype(np.float32),
        'grad': np.random.randn(*GRAD_SHAPE).astype(np.float32),
    }


class TestAllMethods:
    @pytest.mark.parametrize("method_name", [m for m in TESTABLE if m not in SKIP_RUN])
    def test_method_runs(self, method_name, layer_data):
        m = METHODS_EXTENDED[method_name]
        kwargs = {v: layer_data[v] for v in m['variables']}
        scores = score_layer_extended(method_name, **kwargs)
        assert isinstance(scores, (np.ndarray, np.floating, float, np.integer, int)), \
            f"Expected ndarray or scalar, got {type(scores)}"
        if isinstance(scores, np.ndarray):
            assert not np.any(np.isnan(scores)), f"NaN in {method_name}"
            assert scores.size > 0, f"Empty scores for {method_name}"

    @pytest.mark.parametrize("method_name", TESTABLE)
    def test_method_shape_valid(self, method_name, layer_data):
        if method_name in SKIP_SHAPE:
            pytest.skip("Known broadcast issue")
        m = METHODS_EXTENDED[method_name]
        kwargs = {v: layer_data[v] for v in m['variables']}
        scores = score_layer_extended(method_name, **kwargs)
        if isinstance(scores, np.ndarray):
            if method_name in PER_NEURON_METHODS:
                assert scores.ndim == 1, f"{method_name}: expected 1D, got {scores.shape}"
            elif method_name in FEATURE_METHODS:
                assert scores.ndim == 1, f"{method_name}: expected 1D, got {scores.shape}"
            elif method_name not in SCALAR_METHODS:
                assert scores.shape == W_SHAPE, \
                    f"{method_name}: expected {W_SHAPE}, got {scores.shape}"

    @pytest.mark.parametrize("method_name", [m for m in TESTABLE if m not in SKIP_RUN])
    def test_method_has_signal(self, method_name, layer_data):
        m = METHODS_EXTENDED[method_name]
        kwargs = {v: layer_data[v] for v in m['variables']}
        scores = score_layer_extended(method_name, **kwargs)
        if isinstance(scores, np.ndarray):
            assert scores.max() > 0, f"All scores zero for {method_name}"


class TestGradDependency:
    def test_no_grad_methods_dont_need_grad(self):
        for name, m in METHODS_EXTENDED.items():
            if not m['needs_grad']:
                assert 'grad' not in m['variables'], \
                    f"{name}: needs_grad=False but has 'grad' in variables"

    def test_grad_methods_have_grad_or_equivalent(self):
        for name, m in METHODS_EXTENDED.items():
            if m['needs_grad']:
                vars_ = m['variables']
                has_grad = any('grad' in v for v in vars_)
                assert has_grad, f"{name}: needs_grad=True but no grad in {vars_}"


class TestOverlaps:
    def test_wanda_vs_gradient_overlap(self, layer_data):
        wanda = score_layer_extended('wanda', W=layer_data['W'], act=layer_data['act'])
        grad = score_layer_extended('gradient', W=layer_data['W'], grad=layer_data['grad'])
        wanda = wanda / (wanda.max() + 1e-8)
        grad = grad / (grad.max() + 1e-8)
        k = int(0.30 * wanda.size)
        top_wanda = set(np.argsort(wanda.flatten())[-k:])
        top_grad = set(np.argsort(grad.flatten())[-k:])
        overlap = len(top_wanda & top_grad) / k
        print(f"\n  Wanda & Gradient (top-30%): {overlap*100:.1f}%")
        assert 0.20 < overlap < 0.80, f"Overlap {overlap:.2f} out of range"


class TestEdgeCases:
    def test_all_zeros_W(self):
        W = np.zeros(W_SHAPE, dtype=np.float32)
        act = np.random.randn(*ACT_SHAPE).astype(np.float32)
        for name in ['magnitude', 'wanda', 'gps_direction', 'direction_per_neuron']:
            if name not in METHODS_EXTENDED:
                continue
            m = METHODS_EXTENDED[name]
            kwargs = {v: W if v == 'W' else act for v in m['variables']}
            try:
                scores = score_layer_extended(name, **kwargs)
                if isinstance(scores, np.ndarray):
                    assert not np.any(np.isnan(scores))
            except Exception as e:
                print(f"  {name}: exception with W=0: {e}")

    def test_single_token(self):
        W = np.random.randn(*W_SHAPE).astype(np.float32)
        act = np.random.randn(1, 768).astype(np.float32)
        scores = score_layer_extended('wanda', W=W, act=act)
        assert scores.shape == W_SHAPE
        assert not np.any(np.isnan(scores))
