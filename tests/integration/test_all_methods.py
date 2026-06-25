"""
Tests pour les 18 méthodes de scoring.
Vérifie que chaque formule APL compile et produit un score valide.
"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '.')
from src.extraction.scorers.methods_extended import METHODS_EXTENDED, score_layer_extended

W_SHAPE = (3072, 768)
ACT_SHAPE = (128, 768)
GRAD_SHAPE = (3072, 768)

# Méthodes qui produisent un score par neurone (1D) au lieu de par connexion (2D)
PER_NEURON_METHODS = [
    'gps_direction', 'contrast_grad', 'contrast_wanda', 'contrast_gradient',
    'cos_w_x', 'cos_w_grad', 'gps_w_x', 'gps_w_grad', 'gps_cube',
]

# Méthodes qui produisent un scalaire
SCALAR_METHODS = ['contrast_X', 'cos_x_grad', 'gps_x_grad', 'union_wanda_grad', 'union_all3']

# Méthodes qui produisent 1D par feature (à transposer)
FEATURE_METHODS = ['x_times_grad']

@pytest.fixture
def layer_data():
    np.random.seed(42)
    return {
        'W': np.random.randn(*W_SHAPE).astype(np.float32),
        'act': np.random.randn(*ACT_SHAPE).astype(np.float32),
        'grad': np.random.randn(*GRAD_SHAPE).astype(np.float32),
    }


class TestAllMethods:
    """Test chaque méthode individuellement."""
    
    @pytest.mark.parametrize("method_name", list(METHODS_EXTENDED.keys()))
    def test_method_runs(self, method_name, layer_data):
        """Vérifie que la méthode s'exécute sans erreur."""
        m = METHODS_EXTENDED[method_name]
        kwargs = {v: layer_data[v] for v in m['variables']}
        
        scores = score_layer_extended(method_name, **kwargs)
        
        # Vérifications de base
        assert isinstance(scores, (np.ndarray, np.floating, float)), \
            f"Expected ndarray or scalar, got {type(scores)}"
        
        if isinstance(scores, np.ndarray):
            assert not np.any(np.isnan(scores)), f"NaN in {method_name}"
            assert not np.any(np.isinf(scores)), f"Inf in {method_name}"
            assert scores.size > 0, f"Empty scores for {method_name}"
    
    @pytest.mark.parametrize("method_name", list(METHODS_EXTENDED.keys()))
    def test_method_shape_valid(self, method_name, layer_data):
        """Vérifie que la forme est cohérente."""
        m = METHODS_EXTENDED[method_name]
        kwargs = {v: layer_data[v] for v in m['variables']}
        
        scores = score_layer_extended(method_name, **kwargs)
        
        if isinstance(scores, np.ndarray):
            if method_name in PER_NEURON_METHODS:
                # Score par neurone : (d_out,)
                assert scores.ndim == 1 and scores.shape[0] == W_SHAPE[0], \
                    f"{method_name}: expected ({W_SHAPE[0]},), got {scores.shape}"
            elif method_name in FEATURE_METHODS:
                # Score par feature : (d_in,)
                assert scores.ndim == 1 and scores.shape[0] == W_SHAPE[1], \
                    f"{method_name}: expected ({W_SHAPE[1]},), got {scores.shape}"
            elif method_name not in SCALAR_METHODS:
                # Score par connexion : (d_out, d_in)
                assert scores.shape == W_SHAPE, \
                    f"{method_name}: expected {W_SHAPE}, got {scores.shape}"
    
    @pytest.mark.parametrize("method_name", list(METHODS_EXTENDED.keys()))
    def test_method_has_signal(self, method_name, layer_data):
        """Vérifie que le score n'est pas tout à zéro."""
        m = METHODS_EXTENDED[method_name]
        kwargs = {v: layer_data[v] for v in m['variables']}
        
        scores = score_layer_extended(method_name, **kwargs)
        
        if isinstance(scores, np.ndarray):
            assert scores.max() > 0, f"All scores zero for {method_name}"


class TestGradDependency:
    """Vérifie que les méthodes sans grad ne demandent pas grad."""
    
    def test_no_grad_methods_dont_need_grad(self):
        for name, m in METHODS_EXTENDED.items():
            if not m['needs_grad']:
                assert 'grad' not in m['variables'], \
                    f"{name}: needs_grad=False but has 'grad' in variables"
    
    def test_grad_methods_need_grad(self):
        for name, m in METHODS_EXTENDED.items():
            if m['needs_grad']:
                assert 'grad' in m['variables'], \
                    f"{name}: needs_grad=True but 'grad' not in variables"


class TestOverlaps:
    """Mesure les overlaps entre méthodes 2D uniquement."""
    
    def test_wanda_vs_gradient_overlap(self, layer_data):
        """Wanda et Gradient devraient avoir ~50% d'overlap."""
        wanda = score_layer_extended('wanda', W=layer_data['W'], act=layer_data['act'])
        grad = score_layer_extended('gradient', W=layer_data['W'], grad=layer_data['grad'])
        
        wanda = wanda / (wanda.max() + 1e-8)
        grad = grad / (grad.max() + 1e-8)
        
        k = int(0.30 * wanda.size)
        top_wanda = set(np.argsort(wanda.flatten())[-k:])
        top_grad = set(np.argsort(grad.flatten())[-k:])
        overlap = len(top_wanda & top_grad) / k
        
        print(f"\n  Wanda ∩ Gradient (top-30%): {overlap*100:.1f}%")
        assert 0.20 < overlap < 0.80, f"Overlap {overlap:.2f} hors plage"


class TestEdgeCases:
    """Tests aux limites."""
    
    def test_all_zeros_W(self):
        """Méthodes sans grad devraient survivre à W=0."""
        W = np.zeros(W_SHAPE, dtype=np.float32)
        act = np.random.randn(*ACT_SHAPE).astype(np.float32)
        
        for name in ['magnitude', 'wanda', 'gps_direction']:
            m = METHODS_EXTENDED[name]
            kwargs = {v: W if v == 'W' else act for v in m['variables']}
            try:
                scores = score_layer_extended(name, **kwargs)
                if isinstance(scores, np.ndarray):
                    assert not np.any(np.isnan(scores)), f"{name}: NaN with W=0"
            except Exception as e:
                print(f"  {name}: exception with W=0: {e}")
    
    def test_single_token(self):
        """Test avec un seul token."""
        W = np.random.randn(*W_SHAPE).astype(np.float32)
        act = np.random.randn(1, 768).astype(np.float32)
        
        scores = score_layer_extended('wanda', W=W, act=act)
        assert scores.shape == W_SHAPE
        assert not np.any(np.isnan(scores))
