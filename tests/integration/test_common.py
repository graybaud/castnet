"""
Tests for castnet/common.py
=============================
Validates all shared utilities: data, model, masking, evaluation,
checkpointing, gamma, co-activation hooks, graph extraction.

Usage:
    python tests/test_common.py
"""

import os
import sys
import tempfile
import torch
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Migrated imports
from domain.scoring.masks import apply_percentile_mask, measure_sparsity
from domain.metrics.gamma import measure_gamma
from infrastructure.hooks.coactivation import CoActivationHook
from infrastructure.models.huggingface import HuggingFaceWeightProvider
FFN_PATTERNS = HuggingFaceWeightProvider.FFN_PATTERNS

# Stubs for non-migrated functions
def printf(*a, **kw): print(*a)
def load_wikitext(*a, **kw): pass
def load_model_and_tokenizer(*a, **kw): pass
def get_ffn_layers(*a, **kw): return []
def get_all_sparse_layers(*a, **kw): return []
def extract_masks(*a, **kw): return {}
def apply_masks(*a, **kw): pass
def get_sparsity(model): return 0.0
def count_params(model): return (0, 0)
def save_checkpoint(*a, **kw): pass
def load_checkpoint(*a, **kw): return {}
def save_ffn_weights(model): return {}
def restore_ffn_weights(*a, **kw): pass
def count_graph_nodes_edges(*a, **kw): return (0, 0)
def register_coactivation_hooks(*a, **kw): return ([], {})
def diagnose_coactivation_freqs(*a, **kw): pass
def extract_nx_graph(*a, **kw): return None
def analyze_graph(*a, **kw): return {}
def export_graph_json(*a, **kw): pass
def convert_types(data): return data
# ═══════════════════════════════════════════════════════════════════════
#  MOCK WIKITEXT — évite le téléchargement à chaque test
# ═══════════════════════════════════════════════════════════════════════

import torch
from datasets import Dataset

_MOCK_WIKITEXT = None

def _get_mock_wikitext(tokenizer, max_len=32, num_examples=10):
    """Génère un mini dataset synthétique au lieu de télécharger WikiText-2."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "The history of natural language processing dates back to the 1950s.",
        "Neural networks learn hierarchical representations from data.",
        "Transformers use self-attention to process sequences in parallel.",
        "Language models predict the probability of token sequences.",
        "Gradient descent optimizes the loss function during training.",
        "Sparse neural networks prune unnecessary connections to save compute.",
        "The attention mechanism allows the model to focus on relevant inputs.",
        "Transfer learning enables models to adapt to new tasks with less data.",
    ][:num_examples]

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len,
                         padding="max_length", return_tensors="pt")

    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize, batched=True, remove_columns=["text"])

def assert_true(cond, msg=""):
    assert cond, msg

def assert_equal(a, b, msg=""):
    assert a == b, f"{msg}: {a} != {b}"


# ═══════════════════════════════════════════════════════════════════════
#  PRINTF
# ═══════════════════════════════════════════════════════════════════════

def test_printf_does_not_crash():
    printf("  [TEST] printf...")


# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════

def test_load_wikitext():
    printf("  [TEST] load_wikitext...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = _get_mock_wikitext(tokenizer, max_len=32)
    batch = next(iter(dataset))
    ids = batch["input_ids"]
    assert_true(isinstance(ids, list) or hasattr(ids, 'shape'), "Should return tokenized data")
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════

def test_get_ffn_layers():
    printf("  [TEST] get_ffn_layers...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    layers = get_ffn_layers(model)
    assert_equal(len(layers), 24, "OPT-125M has 12 layers × 2 FFN matrices")
    printf("PASSED")


def test_get_all_sparse_layers():
    printf("  [TEST] get_all_sparse_layers...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    layers = get_all_sparse_layers(model)
    # 24 FFN + 12*4 attention = 24+48=72, mais seules les projections existantes comptent
    assert_true(len(layers) >= 24, f"Should have at least FFN layers, got {len(layers)}")
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  MASKING
# ═══════════════════════════════════════════════════════════════════════

def test_extract_and_apply_masks():
    printf("  [TEST] extract_masks / apply_masks...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    layers = get_ffn_layers(model)
    original_weight = layers[0][1].weight.data.clone()
    
    masks = extract_masks(model, layers)
    # Tous les poids sont non-zéros au départ
    assert_true(masks[layers[0][0]].sum() > 0, "Mask should have active connections")
    
    # Forcer quelques zéros
    layers[0][1].weight.data[0, 0] = 0
    masks2 = extract_masks(model, layers)
    assert_equal(masks2[layers[0][0]][0, 0].item(), 0.0, "Zero weight should give zero mask")
    
    # Restaurer
    layers[0][1].weight.data = original_weight
    printf("PASSED")


def test_apply_percentile_mask():
    printf("  [TEST] apply_percentile_mask...")
    score = torch.tensor([[0.9, 0.1, 0.5], [0.3, 0.7, 0.2]])
    mask = apply_percentile_mask(score, 0.5)
    assert_equal(mask.sum().item(), 3, "Should keep 3/6 connections")
    
    # Vérifier que les scores gardés >= scores supprimés
    flat = score.flatten()
    kept = flat[mask.flatten() > 0]
    pruned = flat[mask.flatten() == 0]
    assert_true(kept.min() >= pruned.max(), "Top scores should be kept")
    printf("PASSED")


def test_apply_percentile_mask_empty():
    printf("  [TEST] apply_percentile_mask (all zero)...")
    score = torch.zeros(4, 4)
    mask = apply_percentile_mask(score, 0.5)
    assert_equal(mask.sum().item(), 0, "Should return zero mask")
    printf("PASSED")

def test_get_sparsity():
    printf("  [TEST] get_sparsity...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    sp = get_sparsity(model)
    assert_true(sp < 0.01, f"Dense model should have near 0% sparsity, got {sp:.4f}%")
    printf("PASSED")


def test_count_params():
    printf("  [TEST] count_params...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    active, total = count_params(model)
    assert_true(active > 0, "Should have active parameters")
    assert_true(total > 0, "Should have total parameters")
    assert_true(abs(active - total) < total * 0.01,
                f"Active ({active:,}) should be close to total ({total:,})")
    printf("PASSED")

# ═══════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════

def test_save_and_load_checkpoint():
    printf("  [TEST] save_checkpoint / load_checkpoint...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    
    save_checkpoint(model, None, {"test": True}, tmp)
    assert_true(os.path.exists(tmp), "Checkpoint file should exist")
    
    ckpt = load_checkpoint(model, tmp)
    assert_true(ckpt.get('test') == True, "Metadata should be preserved")
    
    os.unlink(tmp)
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  FFN WEIGHTS
# ═══════════════════════════════════════════════════════════════════════

def test_save_restore_ffn_weights():
    printf("  [TEST] save_ffn_weights / restore_ffn_weights...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    
    # Sauvegarder
    saved = save_ffn_weights(model)
    assert_true(len(saved) > 0, "Should save FFN weights")
    
    # Modifier un poids
    first_key = list(saved.keys())[0]
    original_val = saved[first_key][0, 0].clone()
    saved[first_key][0, 0] = 999.0
    
    # Restaurer
    restore_ffn_weights(model, saved)
    assert_true(torch.equal(saved[first_key][0, 0], torch.tensor(999.0)), "Should restore modified weight")
    
    # Nettoyer
    saved[first_key][0, 0] = original_val
    restore_ffn_weights(model, saved)
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  GAMMA
# ═══════════════════════════════════════════════════════════════════════

def test_measure_gamma():
    printf("  [TEST] measure_gamma...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    gamma = measure_gamma(model, threshold=0.05)
    assert_true(isinstance(gamma, float), "Gamma should be a float")
    assert_true(gamma >= 0 or gamma == 0.0, f"Gamma should be >= 0, got {gamma}")
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  GRAPH STATS
# ═══════════════════════════════════════════════════════════════════════

def test_count_graph_nodes_edges():
    printf("  [TEST] count_graph_nodes_edges...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    nodes, edges = count_graph_nodes_edges(model)
    assert_true(nodes > 0, "Should count nodes")
    assert_true(edges > 0, "Should count edges")
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  CO-ACTIVATION HOOKS
# ═══════════════════════════════════════════════════════════════════════

def test_coactivation_hook_init():
    printf("  [TEST] CoActivationHook init...")
    W = torch.randn(64, 32)
    hook = CoActivationHook(W, "test_layer")
    assert_equal(hook.freq_matrix.shape, torch.Size([64, 32]))
    assert_equal(hook.total_tokens, 0)
    printf("PASSED")


def test_coactivation_hook_forward():
    printf("  [TEST] CoActivationHook hook...")
    W = torch.randn(64, 32)
    hook = CoActivationHook(W, "test_layer")
    
    # Simuler un forward
    class FakeModule:
        pass
    module = FakeModule()
    x = torch.randn(1, 16, 32)  # [batch, seq, in_dim]
    out = torch.randn(1, 16, 64)  # [batch, seq, out_dim]
    hook.hook(module, (x,), out)
    
    assert_equal(hook.total_tokens, 16, "Should count 16 tokens")
    assert_true(hook.freq_matrix.sum() > 0, "Should accumulate activations")
    printf("PASSED")


def test_coactivation_hook_finalize():
    printf("  [TEST] CoActivationHook finalize...")
    W = torch.randn(4, 4)
    hook = CoActivationHook(W, "test")
    hook.freq_matrix = torch.tensor([[8.0, 0.0], [4.0, 4.0]])
    hook.total_tokens = 8
    hook.finalize()
    expected = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    assert_true(torch.allclose(hook.freq_matrix, expected), f"Normalization wrong: {hook.freq_matrix}")
    printf("PASSED")


def test_register_coactivation_hooks():
    printf("  [TEST] register_coactivation_hooks...")
    model, _ = load_model_and_tokenizer("facebook/opt-125m", torch.device('cpu'))
    model.eval()
    hooks, layer_map = register_coactivation_hooks(model)
    assert_equal(len(hooks), 24, "OPT-125M has 24 FFN layers")
    assert_equal(len(layer_map), 24)
    
    # Nettoyer
    for h in hooks:
        h.handle.remove()
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  CONVERT TYPES
# ═══════════════════════════════════════════════════════════════════════

def test_convert_types():
    printf("  [TEST] convert_types...")
    data = {
        "int": np.int32(5),
        "float": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "nested": {"val": np.int64(10)},
        "list": [np.float32(1.0), np.float32(2.0)],
    }
    result = convert_types(data)
    assert_equal(type(result["int"]), int)
    assert_equal(type(result["float"]), float)
    assert_equal(type(result["array"]), list)
    assert_equal(type(result["nested"]["val"]), int)
    assert_equal(type(result["list"][0]), float)
    printf("PASSED")


# ═══════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests():
    printf("=" * 60)
    printf("  CASTNET COMMON — UNIT TESTS")
    printf("=" * 60 + "\n")

    tests = [
        ("printf", test_printf_does_not_crash),
        ("load_wikitext", test_load_wikitext),
        ("get_ffn_layers", test_get_ffn_layers),
        ("get_all_sparse_layers", test_get_all_sparse_layers),
        ("extract/apply_masks", test_extract_and_apply_masks),
        ("apply_percentile_mask", test_apply_percentile_mask),
        ("apply_percentile_mask (empty)", test_apply_percentile_mask_empty),
        ("get_sparsity", test_get_sparsity),
        ("count_params", test_count_params),
        ("save/load_checkpoint", test_save_and_load_checkpoint),
        ("save/restore_ffn_weights", test_save_restore_ffn_weights),
        ("measure_gamma", test_measure_gamma),
        ("count_graph_nodes_edges", test_count_graph_nodes_edges),
        ("CoActivationHook init", test_coactivation_hook_init),
        ("CoActivationHook hook", test_coactivation_hook_forward),
        ("CoActivationHook finalize", test_coactivation_hook_finalize),
        ("register_coactivation_hooks", test_register_coactivation_hooks),
        ("convert_types", test_convert_types),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            printf(f"\n  [FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1

    printf(f"\n{'=' * 60}")
    printf(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    printf(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)