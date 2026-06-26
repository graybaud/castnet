"""Smoke tests for score accumulators — verifies they don't crash."""

import torch
import pytest


class MockModule:
    def __init__(self, out_dim, in_dim):
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight.grad = None  # Will be set by backward

    def register_forward_hook(self, hook_fn):
        class FakeHook:
            def remove(self):
                pass
        return FakeHook()


class MockModel:
    def __init__(self, layers):
        self._layers = layers

    def zero_grad(self):
        for _, m in self._layers:
            m.weight.grad = None

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, ids, labels=None):
        # Simple forward: sum of all layer weights * input mean
        x = ids.float().mean()
        loss = torch.tensor(0.0, requires_grad=True)
        for name, m in self._layers:
            loss = loss + (m.weight * x).sum()
        return type('obj', (), {'loss': loss})()


class MockDataset:
    def __init__(self, num_batches=5):
        self._num = num_batches

    def __iter__(self):
        self._count = 0
        return self

    def __next__(self):
        if self._count >= self._num:
            raise StopIteration
        self._count += 1
        return {'input_ids': torch.randint(0, 1000, (1, 32))}


@pytest.fixture
def mock_layers():
    return [("fc1", MockModule(8, 4)), ("fc2", MockModule(4, 8))]


@pytest.fixture
def mock_model(mock_layers):
    return MockModel(mock_layers)


@pytest.fixture
def mock_dataset():
    return MockDataset(num_batches=2)


class TestAccumulators:

    def test_magnitude_runs(self, mock_model, mock_layers, mock_dataset):
        from infrastructure.scoring.accumulators import accumulate_magnitude_scores
        scores = accumulate_magnitude_scores(mock_model, mock_layers, mock_dataset, 2, 'cpu')
        assert len(scores) == 2
        for name in scores:
            assert scores[name].max() == 1.0

    def test_gradient_runs(self, mock_model, mock_layers, mock_dataset):
        from infrastructure.scoring.accumulators import accumulate_gradient_scores
        scores = accumulate_gradient_scores(mock_model, mock_layers, mock_dataset, 2, 'cpu')
        assert len(scores) == 2
        for name in scores:
            assert not torch.isnan(scores[name]).any()

    def test_softmax_gradient_runs(self, mock_model, mock_layers, mock_dataset):
        from infrastructure.scoring.accumulators import accumulate_softmax_gradient_scores
        scores = accumulate_softmax_gradient_scores(mock_model, mock_layers, mock_dataset, 2, 'cpu')
        assert len(scores) == 2
        for name in scores:
            assert not torch.isnan(scores[name]).any()
