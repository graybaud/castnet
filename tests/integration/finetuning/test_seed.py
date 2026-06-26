"""Tests for set_seed — migrated from legacy finetuning/test_seed.py."""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility — migrated from src.utils."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
