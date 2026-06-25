"""
test_seed.py — Tests for set_seed.
"""

import random
import numpy as np
import torch
from src.finetuning import set_seed


class TestSetSeed:

    def test_reproducibility_python(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_reproducibility_numpy(self):
        set_seed(42)
        a = np.random.rand()
        set_seed(42)
        b = np.random.rand()
        assert a == b

    def test_reproducibility_torch(self):
        set_seed(42)
        a = torch.rand(1).item()
        set_seed(42)
        b = torch.rand(1).item()
        assert a == b
