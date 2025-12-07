# Nōkai Neuromorphic Architecture

## Vue d'ensemble

L'architecture neuromorphique de Nōkai implémente un cerveau artificiel complet basé sur les principes biologiques du cerveau humain.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      NŌKAI NEUROMORPHIC BRAIN v0.2.0                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   INPUT (Tokens)                                                          │
│        │                                                                  │
│        ▼                                                                  │
│   ┌─────────────┐        ┌────────────────────┐                          │
│   │  THALAMUS   │───────▶│ ATTENTION CONTROLLER│ (Ressource Allocation)  │
│   │  (Gateway)  │        └────────────────────┘                          │
│   └──────┬──────┘                 │                                       │
│          │ Filtered               │ energy_check()                        │
│          ▼                        ▼                                       │
│   ┌─────────────┐  ◀─────  ┌────────────────┐                            │
│   │   CORTEX    │  ─────▶  │ WORKING MEMORY │ (Cortex Préfrontal)        │
│   │  (Columns)  │          │     (PFC)      │                            │
│   └──────┬──────┘          └───────┬────────┘                            │
│          │                         │                                      │
│          ▼                         ▼                                      │
│   ┌─────────────┐          ┌────────────────┐                            │
│   │ HIPPOCAMPUS │◀────────▶│SEMANTIC MEMORY │ (Néocortex Long-Terme)     │
│   │ (Épisodique)│  SLEEP   │  (Knowledge)   │                            │
│   └──────┬──────┘  TRANSFER└───────┬────────┘                            │
│          │           │             │                                      │
│          │     ┌─────┴─────┐       │                                      │
│          │     │CONSOLIDATION│      │                                      │
│          │     │  (Sleep)   │      │                                      │
│          │     └───────────┘       │                                      │
│          │                         │                                      │
│          ▼                         ▼                                      │
│   ┌─────────────────────────────────────────┐                            │
│   │           METACOGNITION (dACC)           │ Uncertainty Monitor        │
│   └───────────────────┬─────────────────────┘                            │
│                       │                                                   │
│                       ▼                                                   │
│   ┌─────────────────────────────────────────┐                            │
│   │          DOPAMINE CIRCUIT (VTA)          │ Reward Prediction          │
│   │              dopamine_level              │ RPE Computation            │
│   └───────────────────┬─────────────────────┘                            │
│                       │                                                   │
│                       ▼                                                   │
│   ┌─────────────────────────────────────────┐                            │
│   │             STRIATUM                     │ Action Selection           │
│   │      (Direct/Indirect Pathways)          │ Cost/Benefit Analysis      │
│   └───────────────────┬─────────────────────┘                            │
│                       │                                                   │
│                       ▼                                                   │
│                    OUTPUT                                                 │
│                                                                           │
│   ═══════════════════════════════════════════════════════════════════   │
│   │                    OSCILLATIONS (Theta/Gamma)                    │   │
│   │           Coordinate all modules via phase synchronization       │   │
│   ═══════════════════════════════════════════════════════════════════   │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Les 10 Modules Neuromorphiques

### 1. Thalamus (Passerelle Sensorielle)
**Fichier:** `nokai/thalamus/__init__.py`

```python
from nokai.thalamus import ThalamusGateway

# Filtre les inputs - seuls les tokens pertinents passent
# Implémente l'attention sparse O(n log n) au lieu de O(n²)
thalamus = ThalamusGateway(
    input_dim=256,
    sparsity_target=0.05,  # Seuls 5% des tokens passent
)
```

**Fonctionnalités Bio-inspirées:**
- `energy_check()` - Active seulement si nécessaire
- `synaptic_weights` - Voies fréquentes renforcées
- Oscillation gate (rythme Alpha)

---

### 2. Cortex (Colonnes Corticales)
**Fichier:** `nokai/cortex/`

Traitement hiérarchique via colonnes corticales inspirées du néocortex.

**Fonctionnalités:**
- Activation sparse (10% des neurones actifs)
- Connections latérales entre colonnes
- Prédiction top-down (Predictive Coding)

---

### 3. Mémoire de Travail (Cortex Préfrontal)
**Fichier:** `nokai/prefrontal/__init__.py`

```python
from nokai.prefrontal import PrefrontalWorkingMemory

# Buffer actif avec capacité limitée (7±2 slots)
wm = PrefrontalWorkingMemory(
    dim=256,
    capacity=8,
)
```

**Fonctionnalités Bio-inspirées:**
- Input/Output/Forget gates (comme LSTM mais bio-inspiré)
- Decay sans maintenance active
- Dopamine module le taux de decay

---

### 4. Encodeur Épisodique (Hippocampe)
**Fichier:** `nokai/hippocampus/memory.py`

Stockage vectoriel rapide des expériences avec pattern separation (DG) et pattern completion (CA3).

---

### 5. Mémoire Sémantique (Néocortex)
**Fichier:** `nokai/memory/semantic.py`

```python
from nokai.memory import SemanticMemory

# Base de connaissances à long terme
# Mise à jour LENTE (consolidation)
semantic = SemanticMemory(
    embedding_dim=256,
    max_concepts=1_000_000,
    update_rate=0.001,  # Très lent
)
```

**Efficacité:**
- Memory-mapped pour milliards d'entrées
- HNSW index pour O(log N) retrieval

---

### 6. Moteur de Récompense (VTA/Dopamine)
**Fichier:** `nokai/limbic/dopamine.py`

```python
from nokai.limbic import DopamineCircuit

# Calcule le RPE (Reward Prediction Error)
dopamine = DopamineCircuit(state_dim=256)

# Le niveau de dopamine influence:
# - Taux d'apprentissage (plus haut = plus rapide)
# - Allocation d'attention
# - Priorité de stockage mémoire
```

**Variable clé:** `dopamine_level` ∈ [0, 1]

---

### 7. Sélecteur de Décision (Striatum)
**Fichier:** `nokai/limbic/striatum.py`

```python
from nokai.limbic import StriatumSelector

striatum = StriatumSelector(
    state_dim=256,
    action_dim=256,
)

# Voies directe (GO) vs indirecte (NO-GO)
# Modulé par dopamine:
# - High DA → bias GO (action)
# - Low DA → bias NO-GO (inhibition)
```

---

### 8. Moniteur Métacognitif (dACC)
**Fichier:** `nokai/limbic/dacc.py`

```python
from nokai.limbic import MetacognitiveMonitor

dacc = MetacognitiveMonitor(state_dim=256)

# Évalue:
# - confidence: Certitude de la réponse
# - conflict: Réponses en compétition?
# - error_likelihood: Risque d'erreur
```

**États cognitifs:**
- `AUTOMATIC` - Réponse rapide
- `MONITORING` - Attention modérée
- `DELIBERATE` - Traitement approfondi
- `ERROR_RECOVERY` - Reconsidérer

---

### 9. Contrôleur d'Attention
**Fichier:** `nokai/attention/__init__.py`

```python
from nokai.attention import AttentionController

controller = AttentionController(state_dim=256)

# Alloue dynamiquement les ressources
allocation, meta = controller(state, dopamine_level=0.7)

# Chaque module reçoit une fraction [0, 1]
print(allocation.cortex)      # 0.8
print(allocation.hippocampus) # 0.3
```

---

### 10. Consolidation (Sommeil)
**Fichier:** `nokai/memory/consolidation.py`

```python
from nokai.memory import ConsolidationSystem

consolidation = ConsolidationSystem(embedding_dim=256)

# Simule le sommeil
stats = brain.consolidate(max_steps=100)

# Transfert: Hippocampe → Mémoire Sémantique
# Homéostasie: Normalisation des poids synaptiques
```

---

## Mécanismes Bio-Inspirés Clés

### 1. Plasticité Synaptique (`synaptic_weights`)

Chaque module possède un buffer `synaptic_weights` qui:
- **Augmente** quand une donnée est fréquemment utilisée (LTP)
- **Diminue** via l'homéostasie pendant la consolidation (LTD)

```python
# Exemple dans Striatum
self.register_buffer('synaptic_weights', torch.ones(num_action_candidates))

# Renforcement après succès
self.synaptic_weights[selected_idx] *= 1.01  # LTP

# Homéostasie pendant le sommeil
weights.mul_(scale_factor)  # Global downscaling
```

### 2. Efficacité Énergétique (`energy_check()`)

Chaque module implémente `energy_check()` pour déterminer s'il doit s'activer:

```python
def energy_check(self, state: torch.Tensor) -> bool:
    """Active seulement si l'input est complexe/nouveau."""
    variance = state.var().item()
    return variance > 0.1  # Seuil d'activation
```

### 3. Modulation Dopaminergique

Le niveau de dopamine influence globalement:

```python
# Taux d'apprentissage
learning_rate *= brain.dopamine_circuit.get_learning_modulation()

# Decay mémoire
decay_rate = 0.95 + 0.04 * dopamine_level  # Plus haut DA = moins de decay

# Sélection d'action (GO vs NO-GO)
modulated_benefits = benefits * (1 + DA_weight * (dopamine - 0.5))
```

---

## Utilisation

### Créer un Cerveau Neuromorphique

```python
from nokai import NeuromorphicBrain, NokaiConfig

# Configuration (presets: nano, micro, mini, base, large)
config = NokaiConfig.base()  # 268M params, 6GB VRAM

# Créer le cerveau
brain = NeuromorphicBrain(config)
brain = brain.to("cuda")
```

### Forward Pass

```python
import torch

input_ids = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")

outputs = brain(
    input_ids,
    labels=input_ids,  # Pour le calcul de la loss
    reward=torch.tensor([0.1]),  # Signal de récompense optionnel
    store_memory=True,  # Stocker dans les systèmes de mémoire
    return_brain_state=True,  # Retourner l'état complet
)

# Accès aux sorties
loss = outputs['loss']
logits = outputs['logits']
brain_state = outputs['brain_state']

print(f"Dopamine: {brain_state.dopamine_level}")
print(f"Confidence: {brain_state.confidence}")
```

### Consolidation (Sommeil)

```python
# À faire périodiquement pendant l'entraînement
stats = brain.consolidate(max_steps=100)

print(f"Consolidated: {stats['total_consolidated']} memories")
print(f"Pruned: {stats['total_pruned']} weak memories")
```

### Génération

```python
prompt = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
generated = brain.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
```

---

## Statistiques et Monitoring

```python
# Plasticité synaptique
plasticity = brain.get_plasticity_stats()
# {'thalamus_sw_mean': 1.02, 'striatum_sw_mean': 1.15, ...}

# Efficacité énergétique
efficiency = brain.get_energy_stats()
# {'attention_efficiency': 0.45, 'thalamus_pass_rate': 0.12}
```

---

## Fichiers Créés

```
nokai/
├── brain.py                    # NeuromorphicBrain (intégration complète)
├── thalamus/
│   └── __init__.py             # ThalamusGateway, ThalamicAttention
├── prefrontal/
│   └── __init__.py             # PrefrontalWorkingMemory
├── limbic/
│   ├── __init__.py             # Package exports
│   ├── dopamine.py             # DopamineCircuit, RewardPredictionError
│   ├── striatum.py             # StriatumSelector, ProceduralSkillMemory
│   └── dacc.py                 # MetacognitiveMonitor
├── memory/
│   ├── __init__.py             # Package exports
│   ├── semantic.py             # SemanticMemory
│   └── consolidation.py        # ConsolidationSystem
├── attention/
│   └── __init__.py             # AttentionController, AdaptiveCompute
└── examples/
    └── train_neuromorphic.py   # Exemple d'entraînement complet
```

---

## Prochaines Étapes Recommandées

1. **Entraînement sur données réelles** - Utiliser le script d'exemple
2. **Tuning des hyperparamètres** - Ajuster les seuils d'activation
3. **Métriques de plasticité** - Analyser l'évolution des poids synaptiques
4. **Optimisation CUDA** - Kernels custom pour les opérations sparse
5. **Extension multi-GPU** - Distribution des modules sur plusieurs GPUs
