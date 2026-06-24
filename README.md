# CastNet v2 — From Neural Graphs to Silicon

Low-cost LLM inference via sparse graph extraction, frozen-topology fine-tuning, and neuromorphic hardware deployment.

## Architecture
orchestration/ Points d'entree (scripts, CLI)
application/ Cas d'usage (coordination)
domain/ Metier pur (regles, algorithmes)
infrastructure/ Adaptateurs techniques

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le detail complet.

## Installation

```bash
# Cloner le repo
git clone git@github.com:graybaud/castnet.git
cd castnet

# Creer un venv
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Installer les dependances
pip install -e ".[dev]"
```
Quickstart
Extraction de scores
```bash
python orchestration/extract.py model=sshleifer/tiny-gpt2 method=wanda num_batches=10 device=cpu
```
Generation de masques
```bash
python orchestration/masks.py scores_path=reports/scores.safetensors keep_fraction=0.3
```
Pipeline complet
```bash
python orchestration/pipeline.py model=sshleifer/tiny-gpt2 device=cpu extract.method=magnitude extract.num_batches=10
```
Strategies disponibles (23)
Strategie	Type	Description
magnitude	Built-in	abs(W)
gradient	Built-in	abs(W) * abs(grad)
wanda	Built-in	abs(W) * norm(X)
gps	Built-in	Direction x Selectivity x Distortion
gps_cube	APL	GPS^3: ratio(W) x ratio(X) x ratio(grad)
gcs	APL	Geometric Complement Scoring
sparsegps	APL	Energy x Unicity
softmax_grad	APL	softmax(abs(W)) x abs(grad)
union_all3	APL	max(Wanda, Gradient, GPS^3)
...	...	14 more APL formulas
Voir CONTRIBUTING.md pour ajouter une strategie.

Tests
```bash
# Tests unitaires (sans GPU, sans modele)
pytest tests/unit/ -v

# Tests d'integration (tiny-gpt2 sur CPU)
pytest tests/integration/ -v

# Tous les tests
pytest tests/ -v
```
203 tests, 86% coverage, 0 GPU required for unit tests.

Projets lies
apl-pruning-lab — Mini APL DSL for mathematical formulas (55 formules)

castnet-legacy — Version 1 (monolithique)

Licence
MIT — Gabriel Raybaud
