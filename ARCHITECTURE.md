# CastNet v2 — Architecture

## Vue d'ensemble
orchestration/ 🔴 Couche 4 — Points d'entrée (scripts, CLI, pipelines)
application/ 🟡 Couche 3 — Cas d'usage (coordination métier + infra)
domain/ 🔵 Couche 2 — Métier pur (règles, algorithmes, ports)
infrastructure/ 🟢 Couche 1 — Adaptateurs techniques (frameworks, I/O)


## Les 4 couches

### 🔵 Domain — Le métier

Ce que le projet **sait**. Aucune dépendance à PyTorch, HuggingFace, ou tout autre framework.

| Dossier | Contenu |
|---------|---------|
| `domain/scoring/` | Stratégies de scoring (Wanda, GPS, Gradient...), masques, ports |
| `domain/rules/` | Règles logiques R1-R6 pour la décision de pruning |
| `domain/metrics/` | Métriques d'évaluation (perplexité, sparsité) |
| `domain/hardware/` | Estimations hardware (yield, surface, thermique, routage, SPICE) |
| `domain/analysis/` | Analyse statistique (corrélation, overlap) |
| `domain/visualization/` | Export de graphes pour visualisation |

**Règle :** Si vous pouvez tester la fonction avec `torch.Tensor` et `pytest` sans rien d'autre, c'est du domain.

### 🟢 Infrastructure — Les adaptateurs

Comment le métier interagit avec le monde extérieur. Chaque module implémente un **Port** défini dans `domain/`.

| Dossier | Contenu |
|---------|---------|
| `infrastructure/models/` | HuggingFaceWeightProvider (implémente `WeightProvider`) |
| `infrastructure/hooks/` | ActivationCollector (implémente `ActivationProvider`) |
| `infrastructure/data/` | WikiTextProvider (implémente `BatchProvider`) |
| `infrastructure/persistence/` | SafetensorsScorePersister, SafetensorsMaskPersister |
| `infrastructure/scoring/` | APLScoringBridge (pont vers `apl-pruning`) |

**Règle :** Si le code contient `AutoModelForCausalLM`, `load_dataset`, `register_forward_hook`, c'est de l'infrastructure.

### 🟡 Application — Les cas d'usage

Coordonne `domain` et `infrastructure` pour réaliser un scénario complet.

| Fichier | Cas d'usage |
|---------|-------------|
| `extract_scores.py` | Extraire les scores d'importance d'un modèle |
| `generate_masks.py` | Générer des masques binaires à partir des scores |
| `finetune.py` | Fine-tuner un modèle élagué (frozen topology) |
| `evaluate.py` | Évaluer la perplexité et la sparsité |
| `pipeline.py` | Pipeline complet : extraction → masques → finetuning → évaluation |

### 🔴 Orchestration — Les points d'entrée

Scripts de lancement, CLI, pipelines. Ne contient **aucune logique métier**.

| Fichier | Rôle |
|---------|------|
| `extract.py` | Lancement de l'extraction (Hydra) |
| `masks.py` | Lancement de la génération de masques |
| `pipeline.py` | Lancement du pipeline complet |

## Principes architecturaux

1. **Ports & Adapters (Hexagonal)**
   - Les interfaces (Ports) sont dans `domain/scoring/ports.py`
   - Les implémentations (Adapters) sont dans `infrastructure/`
   - Le domaine ne dépend JAMAIS de l'infrastructure

2. **Injection de dépendances**
   - Chaque use case reçoit ses dépendances au constructeur
   - Pas d'imports cachés, pas de singletons

3. **Formules APL**
   - Les méthodes de scoring sont exprimées en formules mathématiques via `apl-pruning`
   - 55 formules disponibles, 23 stratégies dans CastNet
   - Ajouter une méthode = ajouter une formule, pas du code

## Dépendances entre couches
orchestration → application → domain (ports)
→ infrastructure (adaptateurs)
infrastructure → domain (ports) ← dépendance inversée !


`infrastructure` dépend de `domain` (elle implémente les ports).
`domain` ne dépend de **rien**.
