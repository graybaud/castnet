# CastNet v2 — From Neural Graphs to Silicon

## Architecture

| Dossier | Couche | Rôle |
|---------|--------|------|
| domain/ | Métier | Règles pures, algorithmes, Ports |
| infrastructure/ | Technique | Adaptateurs (HuggingFace, hooks, etc.) |
| application/ | Use cases | Coordination métier + infrastructure |
| orchestration/ | Lancement | Scripts, CLI, pipelines |

## Installation

Installer en mode développement :
  pip install -e ".[dev]"

## Quickstart

  python orchestration/extract.py model=microsoft/phi-2 method=wanda
