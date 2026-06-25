"""Use Case : Evaluate a sparse model."""

from dataclasses import dataclass
import math
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from domain.scoring.ports import WeightProvider, MaskPersister
from domain.scoring.masks import count_sparsity


@dataclass
class EvalResult:
    perplexity: float
    sparsity: dict


class EvaluateUseCase:
    """Evaluates perplexity on WikiText-2 validation set (contiguous blocks)."""

    def __init__(
        self,
        model: WeightProvider,
        mask_persister: MaskPersister,
        tokenizer_name: str = "facebook/opt-125m",
    ):
        self.model = model
        self.mask_persister = mask_persister
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._dense_weights = {
            name: model.get_weight(name).clone()
            for name in model.layer_names()
        }

    def execute(
        self,
        mask_path: str,
        block_size: int = 512,
        max_tokens: int = 50000,
    ) -> EvalResult:
        masks = self.mask_persister.load(mask_path)
        sparsity = count_sparsity(masks)

        # Restore dense weights + apply mask
        for name, w in self._dense_weights.items():
            self.model.get_weight(name).data.copy_(w)
        for name, mask in masks.items():
            w = self.model.get_weight(name)
            w.data = w.data * mask.to(w.dtype)

        # Load WikiText-2 validation as contiguous text
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
        text = ' '.join([x['text'] for x in dataset if len(x['text'].strip()) > 10])
        tokens = self.tokenizer.encode(text, return_tensors='pt')[0]

        # Evaluate on contiguous blocks
        self.model.model.eval()
        total_loss = 0.0
        total_tok = 0
        device = self.model.model.device

        with torch.no_grad():
            for i in range(0, min(len(tokens) - block_size, max_tokens), block_size):
                block = tokens[i:i + block_size].unsqueeze(0).to(device)
                out = self.model.model(block, labels=block)
                total_loss += out.loss.item() * block_size
                total_tok += block_size

        perplexity = math.exp(total_loss / total_tok) if total_tok > 0 else float('inf')
        return EvalResult(perplexity=perplexity, sparsity=sparsity)
