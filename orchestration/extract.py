"""Point d'entree — Extraction de scores."""

import argparse
import torch
from domain.scoring.strategies import get_strategy
from domain.scoring.masks import count_sparsity
from infrastructure.models.huggingface import HuggingFaceWeightProvider
from infrastructure.data.providers import WikiTextProvider
from infrastructure.hooks.activation_collector import ActivationCollector
from infrastructure.persistence.safetensors_persister import SafetensorsScorePersister
from application.extract_scores import ExtractScoresUseCase


def main():
    parser = argparse.ArgumentParser(description="CastNet — Extraction de scores")
    parser.add_argument("--model", type=str, default="microsoft/phi-2")
    parser.add_argument("--method", type=str, default="wanda",
                       choices=["magnitude", "gradient", "wanda", "gps"])
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--output", type=str, default="reports/scores.safetensors")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model} | Method: {args.method}")

    # Infrastructure
    model = HuggingFaceWeightProvider(args.model, device)
    dataset = WikiTextProvider(args.model, args.max_len)

    # Trouver les vrais noms de modules (avec des points, pas des underscores)
    real_names = list(dict(model._model.named_modules()).keys())
    ffn_names = [
        n for n in real_names
        if any(p in n.lower() for p in HuggingFaceWeightProvider.FFN_PATTERNS)
    ]
    collector = ActivationCollector(model._model, ffn_names)
    persister = SafetensorsScorePersister()

    # Metier
    strategy = get_strategy(args.method)

    # Application
    use_case = ExtractScoresUseCase(
        strategy=strategy,
        model=model,
        dataset=dataset,
        collector=collector,
        persister=persister,
    )

    # Execution
    result = use_case.execute(
        num_batches=args.num_batches,
        output_path=args.output,
    )

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION TERMINEE")
    print(f"  Modele   : {args.model}")
    print(f"  Methode  : {args.method}")
    print(f"  Couches  : {result.num_layers}")
    print(f"  Batches  : {result.num_batches}")
    print(f"  Sortie   : {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
