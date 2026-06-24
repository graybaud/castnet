# Contribuer à CastNet v2

## Ajouter une nouvelle stratégie de scoring

### Niveau 1 : Formule simple (1 ligne)

Si votre stratégie peut s'exprimer en une formule mathématique simple :

1. Ajoutez la formule dans `apl-pruning-lab/apl_pruning/scorers.py` :

```python
METHODS.update({
    "ma_strategie": {
        "formula": "|W| x log(|grad| + 1)",
        "needs_grad": True,
        "description": "Ma strategie: |W| x log(|grad| + 1)",
        "variables": ["W", "grad"],
    },
})
```
Ajoutez l'alias dans domain/scoring/apl_strategies.py :

```python
APL_FORMULAS = {
    ...
    "ma_strategie": "ma_strategie",
}
```
C'est tout. La stratégie est disponible via get_strategy("ma_strategie").

Niveau 2 : Formule multi-lignes (APL)
Si votre stratégie nécessite des étapes intermédiaires :

```python
METHODS.update({
    "ma_strategie_complexe": {
        "formula": """
            a <- max(|W|, dim=-1) / mean(|W|, dim=-1)
            b <- max(|grad|, dim=-1) / mean(|grad|, dim=-1)
            a x b x mean(|act|, dim=0)
        """,
        "needs_grad": True,
        "description": "Produit des ratios W et grad, pondere par activation",
        "variables": ["W", "grad", "act"],
    },
})
```
Niveau 3 : Code Python (si nécessaire)
Si la formule est trop complexe pour APL :

Créez une classe dans domain/scoring/strategies.py :

```python
class MaStrategie(ScoringStrategy):
    """Description de ma strategie."""

    def calculate(self, weights, activations, layer_name):
        W = weights.get_weight(layer_name)
        X = activations.get_input_activations(layer_name)
        # Votre calcul ici
        scores = ...
        s_max = scores.max()
        return scores / s_max if s_max > 0 else scores
```
Enregistrez-la :

```python
STRATEGY_REGISTRY["ma_strategie"] = MaStrategie
```
Ajoutez les tests dans tests/unit/test_strategies.py.

Ajouter un nouveau modèle
Créez un adaptateur dans infrastructure/models/ :

```python
class MonModelProvider(WeightProvider):
    def __init__(self, model_name, device):
        self.model = load_model(model_name)
        self._ffn_layers = self._detect_ffn_layers()

    def _detect_ffn_layers(self):
        # Votre logique de détection
        ...

    def get_weight(self, layer_name):
        return self._ffn_layers[layer_name].weight.data.float()

    # ... autres méthodes du port
```
Ajoutez les patterns FFN dans FFN_PATTERNS si nécessaire.

Ajouter un nouveau dataset
Implémentez BatchProvider dans infrastructure/data/providers.py :

```python
class C4Provider(BatchProvider):
    def get_batches(self, num_batches, batch_size=1):
        # Votre logique de chargement
        ...
```
Ajouter une nouvelle règle logique
Ajoutez la fonction dans domain/rules/logical_rules.py :

```python
def rule_r7_ma_regle(mon_score):
    return mon_score > mon_seuil
```
Ajoutez les tests dans tests/unit/test_rules.py.

Ajouter une métrique d'évaluation
Ajoutez la fonction dans domain/metrics/ :

```python
def compute_ma_metrique(logits, targets):
    # Votre calcul
    ...
```
Structure des tests
```text
tests/
├── unit/                    # Tests sans GPU, sans modèle réel
│   ├── test_strategies.py   # Tests des stratégies
│   ├── test_rules.py        # Tests des règles
│   ├── test_metrics.py      # Tests des métriques
│   └── ...
├── integration/             # Tests avec tiny-gpt2 sur CPU
│   ├── test_all_strategies.py
│   └── test_hydra_*.py
└── e2e/                     # Tests bout-en-bout (futur)
```
Checklist pour une PR
La nouvelle fonction est dans le bon dossier (domain, infrastructure, application, orchestration)

Le domaine ne dépend PAS de transformers, datasets, torch.nn

Les tests unitaires passent : pytest tests/unit/

Les tests d'intégration passent : pytest tests/integration/

Tous les tests passent : pytest tests/

Les commentaires et messages sont en anglais

La stratégie est enregistrée dans le STRATEGY_REGISTRY ou APL_FORMULAS
