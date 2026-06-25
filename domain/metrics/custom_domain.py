"""Custom domain evaluation — Hardware/ML knowledge questions."""

CUSTOM_DOMAIN_QUESTIONS = [
    # Hardware / Electronics
    ("What is the typical supply voltage for 130nm CMOS?", "1.2V"),
    ("What is Kirchhoff's current law?", "The sum of currents entering a node equals the sum leaving"),
    ("What is the bandgap of silicon at room temperature?", "1.12 eV"),
    ("What is Moore's law?", "Transistor count doubles approximately every two years"),
    ("What is the difference between SRAM and DRAM?", "SRAM is faster and more expensive, DRAM is denser and cheaper"),
    # Machine Learning
    ("What is backpropagation?", "Algorithm for computing gradients in neural networks"),
    ("What is the softmax function?", "Normalizes a vector into a probability distribution"),
    ("What is a Transformer in machine learning?", "Neural network architecture using self-attention"),
    ("What is pruning in neural networks?", "Removing unnecessary connections to reduce model size"),
    ("What is perplexity in language modeling?", "Exponential of cross-entropy loss, measures prediction uncertainty"),
    # Physics
    ("What is Ohm's law?", "V = I x R, voltage equals current times resistance"),
    ("What is a semiconductor?", "Material with conductivity between conductor and insulator"),
    ("What is quantum tunneling?", "Particle passing through a potential barrier"),
    ("What is the speed of light?", "3 x 10^8 m/s"),
    ("What is the second law of thermodynamics?", "Entropy of an isolated system always increases"),
]


def evaluate_custom_domain(
    model,
    tokenizer,
    questions: list[tuple[str, str]] | None = None,
) -> dict:
    """Evaluate a model on custom domain questions.

    Args:
        model: Model with a generate() method.
        tokenizer: Tokenizer with encode() and decode().
        questions: List of (question, expected_answer) tuples.

    Returns:
        dict with accuracy and per-question results.
    """
    if questions is None:
        questions = CUSTOM_DOMAIN_QUESTIONS

    correct = 0
    results = []

    for question, expected in questions:
        prompt = f"Question: {question}\nAnswer:"
        ids = tokenizer.encode(prompt, return_tensors="pt")
        out = model.generate(ids, max_new_tokens=20, do_sample=False)
        answer = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

        # Simple keyword match
        expected_words = expected.lower().split()[:3]
        is_correct = any(word.lower() in answer.lower() for word in expected_words)

        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "expected": expected,
            "answer": answer[:200],
            "correct": is_correct,
        })

    return {
        "accuracy": correct / len(questions),
        "n_questions": len(questions),
        "correct": correct,
        "per_question": results,
    }
