"""Qualitative generation tests — Pure domain logic."""

from typing import Protocol


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface for qualitative tests."""
    def encode(self, text: str, return_tensors: str = "pt"): ...
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str: ...


class ModelProtocol(Protocol):
    """Minimal model interface for qualitative tests."""
    def generate(self, input_ids, max_new_tokens: int, do_sample: bool, num_beams: int, temperature: float = 1.0, top_p: float = 1.0): ...


QUALITATIVE_PROMPTS = [
    "The capital of France is",
    "Machine learning is a subset of",
    "The theory of relativity was developed by",
    "The meaning of life is",
    "To solve this problem, we need to",
]


def run_greedy_tests(model: ModelProtocol, tokenizer: TokenizerProtocol) -> list[str]:
    """Run greedy generation tests and return results."""
    results = []
    for prompt in QUALITATIVE_PROMPTS:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        out = model.generate(ids, max_new_tokens=30, do_sample=False, num_beams=1)
        results.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return results


def run_sampled_tests(model: ModelProtocol, tokenizer: TokenizerProtocol) -> list[str]:
    """Run sampled generation tests and return results."""
    results = []
    for prompt in QUALITATIVE_PROMPTS:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        out = model.generate(
            ids, max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.9
        )
        results.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return results


def analyze_repetition(text: str) -> dict:
    """Analyze word repetition in generated text."""
    from collections import Counter
    words = text.split()
    if len(words) < 10:
        return {"unique_ratio": 1.0, "most_common": []}
    wc = Counter(words).most_common(5)
    return {
        "unique_ratio": len(set(words)) / len(words),
        "most_common": wc,
        "total_words": len(words),
    }
