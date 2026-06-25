"""Output stability — Compare greedy generation between dense and sparse models."""

from typing import Protocol


class ModelProtocol(Protocol):
    def generate(self, input_ids, max_new_tokens: int, do_sample: bool, num_beams: int): ...


class TokenizerProtocol(Protocol):
    def encode(self, text: str, return_tensors: str = "pt"): ...
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str: ...


DEFAULT_STABILITY_PROMPTS = [
    "The capital of France is",
    "Machine learning is a subset of",
    "The theory of relativity was developed by",
    "The meaning of life is",
    "To solve this problem, we need to",
    "The main advantage of this approach is",
    "Climate change can be addressed by",
    "The difference between DNA and RNA is",
]


def compute_stability(
    model_sparse,
    model_dense,
    tokenizer,
    prompts: list[str] | None = None,
    max_new_tokens: int = 50,
) -> dict:
    """Compare greedy outputs between sparse and dense models.

    Returns:
        dict with exact_match_rate and per-prompt results.
    """
    if prompts is None:
        prompts = DEFAULT_STABILITY_PROMPTS

    exact_matches = 0
    per_prompt = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        out_s = model_sparse.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        out_d = model_dense.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)

        text_s = tokenizer.decode(out_s[0], skip_special_tokens=True)
        text_d = tokenizer.decode(out_d[0], skip_special_tokens=True)

        is_exact = text_s == text_d
        if is_exact:
            exact_matches += 1

        per_prompt.append({
            "prompt": prompt,
            "sparse_output": text_s[:200],
            "dense_output": text_d[:200],
            "exact_match": is_exact,
        })

    return {
        "exact_match_rate": exact_matches / len(prompts),
        "total_prompts": len(prompts),
        "per_prompt": per_prompt,
    }
