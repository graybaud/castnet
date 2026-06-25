"""Orchestration — Model evaluation entry point."""

import hydra
from omegaconf import DictConfig
import math
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from infrastructure.models.huggingface import HuggingFaceWeightProvider
from infrastructure.persistence.safetensors_persister import SafetensorsMaskPersister
from domain.scoring.masks import count_sparsity
from domain.metrics.perplexity import compute_perplexity_from_total


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {cfg.model} | Mode: {cfg.mode}")

    model = HuggingFaceWeightProvider(cfg.model, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    mask_persister = SafetensorsMaskPersister()

    # Save dense weights once
    dense_weights = {
        name: model.get_weight(name).clone()
        for name in model.layer_names()
    }

    # Load validation data once
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
    text = ' '.join([x['text'] for x in dataset if len(x['text'].strip()) > 10])
    tokens = tokenizer.encode(text, return_tensors='pt')[0]

    def eval_model(mask_path=None):
        """Evaluate with optional mask. Returns (perplexity, sparsity_pct)."""
        # Restore dense weights
        for name, w in dense_weights.items():
            model.get_weight(name).data.copy_(w)

        sparsity_pct = 0.0
        if mask_path:
            masks = mask_persister.load(mask_path)
            sp = count_sparsity(masks)
            sparsity_pct = sp['overall_sparsity_pct']
            for name, mask in masks.items():
                w = model.get_weight(name)
                w.data = w.data * mask.to(w.dtype)

        model.model.eval()
        total_loss, total_tok = 0.0, 0
        block_size = 512
        max_tokens = 50000

        with torch.no_grad():
            for i in range(0, min(len(tokens) - block_size, max_tokens), block_size):
                block = tokens[i:i + block_size].unsqueeze(0).to(device)
                out = model.model(block, labels=block)
                total_loss += out.loss.item() * block_size
                total_tok += block_size

        perp = compute_perplexity_from_total(total_loss, total_tok)
        return perp, sparsity_pct

    if cfg.mode == "dense":
        perp, _ = eval_model()
        print(f"Dense perplexity: {perp:.1f}")

    elif cfg.mode == "perplexity":
        perp, sp = eval_model(cfg.mask_path)
        print(f"Perplexity: {perp:.1f} | Sparsity: {sp:.1f}%")

    elif cfg.mode == "all":
        masks_to_eval = [
            ("Dense", None),
            ("Wanda", "reports/opt125m_wanda_masks.safetensors"),
            ("Gradient", "reports/opt125m_gradient_masks.safetensors"),
            ("GPS", "reports/opt125m_gps_masks.safetensors"),
            ("Union", "reports/opt125m_union_gps_grad.safetensors"),
        ]
        print(f"\n{'='*60}")
        print(f"  {'Method':<15} {'Sparsity':>10} {'Perplexity':>12}")
        print(f"  {'-'*45}")
        for name, mask_path in masks_to_eval:
            perp, sp = eval_model(mask_path)
            print(f"  {name:<15} {sp:>9.1f}% {perp:>12.1f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
