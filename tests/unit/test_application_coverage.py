"""Complete coverage for application use cases — evaluate, finetune, pipeline."""

import torch
import pytest
from domain.scoring.ports import WeightProvider, BatchProvider, MaskPersister
from domain.scoring.masks import count_sparsity
from application.evaluate import EvaluateUseCase, EvalResult
from application.finetune import FinetuneUseCase, FinetuneResult
from application.pipeline import PipelineResult, CastNetPipeline
from application.extract_scores import ExtractResult
from application.generate_masks import MaskResult


class FakeWeightProvider(WeightProvider):
    def __init__(self, W):
        self._W = W
        self._call_count = 0
    def get_weight(self, name): return self._W
    def get_bias(self, name): return torch.zeros(self._W.shape[0])
    def get_gradient(self, name): return None
    def layer_names(self): return ["test"]
    def forward_backward(self, batch):
        self._call_count += 1
        return 2.0
    def zero_grad(self): pass


class FakeBatchProvider(BatchProvider):
    def __init__(self, num_batches=10):
        self._num = num_batches
    def get_batches(self, num_batches, batch_size=1):
        for _ in range(min(num_batches, self._num)):
            yield torch.ones(1, 16, dtype=torch.long)


class FakeMaskPersister(MaskPersister):
    def __init__(self, masks=None):
        self._masks = masks or {"test": torch.ones(5, 10)}
        self._saved = None
    def save(self, masks, path): self._saved = (masks, path)
    def load(self, path): return self._masks


class TestEvaluateUseCase:

    def test_execute(self):
        W = torch.randn(5, 10)
        model = FakeWeightProvider(W)
        dataset = FakeBatchProvider(num_batches=5)
        mask_persister = FakeMaskPersister()

        uc = EvaluateUseCase(model, dataset, mask_persister)
        result = uc.execute("dummy.safetensors", num_batches=5)

        assert isinstance(result, EvalResult)
        assert result.perplexity > 0
        assert result.sparsity["total_active"] > 0

    def test_no_batches(self):
        W = torch.randn(5, 10)
        model = FakeWeightProvider(W)
        dataset = FakeBatchProvider(num_batches=0)
        mask_persister = FakeMaskPersister()

        uc = EvaluateUseCase(model, dataset, mask_persister)
        result = uc.execute("dummy.safetensors", num_batches=0)

        assert result.perplexity == float("inf")


class TestFinetuneUseCase:

    def test_execute(self):
        W = torch.randn(5, 10)
        model = FakeWeightProvider(W)
        dataset = FakeBatchProvider(num_batches=5)
        mask_persister = FakeMaskPersister()

        # Mock minimal du modele sous-jacent
        param = torch.nn.Parameter(W.clone())
        model._model = type('obj', (object,), {
            'state_dict': lambda self: {},
            'parameters': lambda self: [param],
        })()

        uc = FinetuneUseCase(model, dataset, mask_persister)
        result = uc.execute(
            "dummy.safetensors",
            "checkpoint.pt",
            epochs=1,
            batches_per_epoch=3,
        )

        assert isinstance(result, FinetuneResult)
        assert result.epochs == 1
        assert result.final_loss > 0


class TestPipelineResult:

    def test_all_fields(self):
        result = PipelineResult(
            extract=ExtractResult(scores={}, num_layers=5, num_batches=100),
            mask=MaskResult(masks={}, sparsity={"overall_sparsity_pct": 50.0}, keep_fraction=0.3),
            finetune=FinetuneResult(checkpoint_path="ckpt.pt", final_loss=1.5, sparsity={}, epochs=3),
            evaluate=EvalResult(perplexity=10.0, sparsity={}),
        )

        assert result.extract.num_layers == 5
        assert result.mask.keep_fraction == 0.3
        assert result.finetune.final_loss == 1.5
        assert result.evaluate.perplexity == 10.0

    def test_optional_steps_none(self):
        result = PipelineResult(
            extract=ExtractResult({}, 0, 0),
            mask=MaskResult({}, {}, 0.5),
            finetune=None,
            evaluate=None,
        )
        assert result.finetune is None
        assert result.evaluate is None
