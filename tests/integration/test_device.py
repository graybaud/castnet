"""
test_device.py — Tests for get_device and get_model_device.
"""

import torch
import torch.nn as nn
from src.extraction.utils.device import get_device, get_model_device


class TestGetDevice:

    def test_cpu(self):
        assert get_device("cpu").type == "cpu"

    def test_cuda(self):
        d = get_device("cuda:0")
        assert d.type == "cuda" and d.index == 0

    def test_none(self):
        assert get_device(None).type in ["cpu", "cuda"]


class TestGetModelDevice:

    def test_cpu_model(self):
        assert get_model_device(nn.Linear(10, 10)).type == "cpu"

    def test_no_device_attr(self):
        assert get_model_device(nn.Linear(10, 10)) is not None

    def test_cuda_if_available(self):
        if torch.cuda.is_available():
            assert get_model_device(nn.Linear(10, 10).cuda()).type == "cuda"
