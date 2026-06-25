"""
test_sparsity.py — Tests for measure_sparsity.
Migrated from legacy. Imports changed.
"""

import torch
from domain.scoring.masks import measure_sparsity


class TestMeasureSparsity:

    def test_fully_dense(self, ffn_model):
        nodes, edges, total, sp = measure_sparsity(ffn_model)
        assert nodes == 4 + 8 + 8 + 4  # 24
        assert edges == 4*8 + 8*4  # 64
        assert total == 64
        assert sp == 0.0

    def test_fully_sparse(self, ffn_model):
        with torch.no_grad():
            ffn_model.fc1.weight.zero_()
            ffn_model.fc2.weight.zero_()
        _, edges, _, sp = measure_sparsity(ffn_model)
        assert edges == 0
        assert sp == 100.0

    def test_half_sparse(self, ffn_model):
        with torch.no_grad():
            ffn_model.fc1.weight.zero_()
        _, edges, total, sp = measure_sparsity(ffn_model)
        assert edges == 8 * 4  # only fc2 active
        assert sp == 50.0

    def test_ignores_non_ffn(self, ffn_model):
        with torch.no_grad():
            ffn_model.attn.weight.zero_()
        nodes, edges, total, sp = measure_sparsity(ffn_model)
        assert nodes == 24
        assert edges == 64

    def test_llama_model(self, llama_model):
        nodes, edges, total, sp = measure_sparsity(llama_model)
        assert nodes == 16+32 + 16+32 + 32+16  # 144
        assert edges == 16*32 + 16*32 + 32*16  # 1536
        assert sp == 0.0
