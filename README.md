# 🧠 Nōkai (脳海) - Architecture Cognitive Biomimétique

<div align="center">

**Le Premier Cerveau Artificiel Véritablement Bio-Inspiré**

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)]()

</div>

---

## 🎯 Vision

Nōkai n'est pas "un autre LLM". C'est une **Architecture Cognitive Biomimétique** qui reproduit les mécanismes fondamentaux du cerveau humain :

| Problème des LLM Actuels | Solution Nōkai |
|-------------------------|----------------|
| 🔴 **Statiques** : N'apprennent plus après l'entraînement | ✅ **Plasticité Synaptique** : Apprentissage Hebbien immédiat |
| 🔴 **Inefficients** : 100% des poids activés (O(N²)) | ✅ **Sparsity Thalamique** : 5% d'activation seulement |
| 🔴 **Sans but** : Minimisation de perte statistique | ✅ **Dopamine Homéostatique** : Récompense basée sur la SURPRISE |
| 🔴 **Character-level** : Manipulent des lettres | ✅ **Tokenization BPE** : Comprennent des CONCEPTS |

---

## 🆕 v0.3.0 - L'Éveil Cognitif

Cette version apporte trois améliorations majeures :

### 1. 📚 Tokenization BPE (Compréhension Sémantique)

```python
from nokai.tokenization import NokaiTokenizer, TokenizerConfig

# Créer et entraîner le tokenizer
tokenizer = NokaiTokenizer.train(
    texts=corpus,
    config=TokenizerConfig(vocab_size=32000)
)

# L'IA manipule maintenant des mots/concepts, pas des lettres !
tokens = tokenizer.encode("Le cerveau utilise la dopamine pour apprendre")
# → [1, 534, 2891, 891, 45, 7823, 234, 892, 2]
```

**Analogie Biologique :** Le cerveau ne traite pas les lettres une par une. Il reconnaît des morphèmes (unités de sens) et des mots entiers.

### 2. 💊 Dopamine Homéostatique (Plus jamais à 1.0)

```python
from nokai.limbic import DopamineCircuit

circuit = DopamineCircuit(state_dim=256)

# La dopamine se base sur la SURPRISE, pas le succès brut
for step in range(100):
    state, meta = circuit(hidden_state, reward=constant_reward)
    print(f"DA: {state.effective_signal:.3f}, Habituation: {meta['habituation']:.3f}")
    # → DA décroît vers 0.5 si la récompense est constante (homéostasie)
```

**Formule Mathématique :**
```
δ(t) = R(t) + γ·V(s_{t+1}) - V(s_t)   # Reward Prediction Error
DA_effective = DA_raw - Baseline + 0.5  # Ajustement homéostatique
```

### 3. ⚡ Apprentissage Hebbien Immédiat

```python
from nokai.learning import HebbianPlasticity, HebbianConfig

hebbian = HebbianPlasticity(
    in_features=256,
    out_features=512,
    config=HebbianConfig(
        learning_rate=0.001,
        dopamine_gating=True,  # Apprend seulement si DA > 0.3
    )
)

# L'apprentissage se fait PENDANT le forward pass !
output = layer(x)
hebbian.apply_update(layer.weight, pre=x, post=output, dopamine=da_level)
```

**Règle de Hebb :** "Les neurones qui s'activent ensemble se connectent plus fortement."

---

## 🏗️ Architecture Neuromorphique

```
┌──────────────────────────────────────────────────────────────┐
│                     NŌKAI NEUROMORPHIC BRAIN                  │
├──────────────────────────────────────────────────────────────┤
│  INPUT → [THALAMUS] → Filtre/Route (5% sparsity)             │
│            ↓                                                  │
│  [CORTEX] ←→ [WORKING MEMORY] ←→ [HIPPOCAMPUS]              │
│      ↓              ↓                   ↓                    │
│  [SEMANTIC] ←── [CONSOLIDATION] ←── [EPISODIC]              │
│            ↓                                                  │
│  [dACC] → Incertitude → [ATTENTION CONTROLLER]               │
│            ↓                                                  │
│  [STRIATUM] ←── [DOPAMINE/VTA] → Sélection d'Action          │
│            ↓                                                  │
│  OUTPUT ← Décision/Réponse                                   │
└──────────────────────────────────────────────────────────────┘
```

### Modules Clés

| Module | Région Cérébrale | Fonction |
|--------|------------------|----------|
| `ThalamusGateway` | Thalamus | Filtrage sensoriel, 5% des tokens passent |
| `Cortex` | Néocortex | Traitement hiérarchique par colonnes corticales |
| `HippocampalMemory` | Hippocampe | Mémoire épisodique, completion de patterns |
| `PrefrontalWorkingMemory` | Cortex Préfrontal | Mémoire de travail, contrôle exécutif |
| `DopamineCircuit` | VTA/NAc | Récompense, motivation, modulation apprentissage |
| `SemanticMemory` | Neocortex | Connaissances à long terme |
| `ConsolidationSystem` | - | Consolidation mémoire ("sommeil") |

---

## 🚀 Démarrage Rapide

### Installation

```bash
git clone https://github.com/JeremGamingYT/Nokai.git
cd Nokai
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .
pip install tokenizers datasets  # Pour BPE et données
```

### Entraînement Cognitif V2

```bash
# Entraînement complet avec toutes les améliorations
python scripts/train_cognitive_v2.py \
    --preset mini \
    --epochs 10 \
    --vocab_size 32000 \
    --hebbian_lr 0.001

# Options pour désactiver des fonctionnalités
python scripts/train_cognitive_v2.py \
    --no_hebbian           # Désactive apprentissage Hebbien
    --no_dopamine_gating   # Désactive gating dopaminergique
    --no_bpe               # Utilise tokenization caractère
```

### Test Rapide

```bash
python scripts/test_cognitive_v2.py
```

---

## 📊 Configurations Disponibles

| Preset | Paramètres | VRAM | Usage |
|--------|-----------|------|-------|
| `nano` | ~4M | 200MB | Tests rapides |
| `micro` | ~17M | 500MB | Prototypage |
| `mini` | ~67M | 2GB | Entraînement léger |
| `base` | ~268M | 6GB | Production (RTX 3060+) |
| `large` | ~1B | 16GB | Haute performance |

---

## 🧬 Principes Biologiques Implémentés

### 1. Sparsité Métabolique
Le cerveau consomme 20W mais traite des informations complexes. Nous reproduisons cette efficacité :
- Seulement 5% des neurones actifs à chaque instant
- `energy_check()` avant chaque module pour décider de l'activation

### 2. Plasticité Synaptique
```
Δw_ij = η · DA · (x_j · x_i - α · x_j² · w_ij)
```
- **Règle de Oja** : Normalisation pour éviter l'explosion des poids
- **BCM** : Seuil glissant pour métaplasticité
- **Dopamine** : Gate l'apprentissage (pas de DA = pas d'apprentissage)

### 3. Oscillations Neurales
- **Theta (4-8 Hz)** : Coordination mémoire-cortex
- **Gamma (30-100 Hz)** : Binding perceptuel
- Les oscillations modulent le traitement à travers les modules

### 4. Consolidation Mémoire ("Sommeil")
Périodiquement, le modèle :
- Rejoue les souvenirs récents
- Transfère les souvenirs importants vers la mémoire sémantique
- Applique l'homéostasie synaptique (downscaling)

---

## 📁 Structure du Projet

```
nokai/
├── __init__.py           # Exports principaux
├── brain.py              # NeuromorphicBrain (intégration complète)
├── model.py              # NokaiModel (version simplifiée)
├── config.py             # Configurations
│
├── cortex/               # Traitement cortical
│   ├── cortex.py         # Assemblage cortical
│   └── column.py         # Colonnes corticales avec Hebbian
│
├── limbic/               # Système limbique
│   ├── dopamine.py       # Circuit dopamine V1 (legacy)
│   ├── dopamine_v2.py    # Circuit dopamine homéostatique ✨
│   ├── striatum.py       # Sélection d'action
│   └── dacc.py           # Métacognition
│
├── learning/             # Règles d'apprentissage
│   ├── hebbian.py        # Hebbien V1 (legacy)
│   ├── hebbian_v2.py     # BCM + Dopamine gating ✨
│   └── predictive.py     # Codage prédictif
│
├── tokenization/         # Tokenization BPE ✨
│   ├── bpe_tokenizer.py  # NokaiTokenizer
│   └── __init__.py
│
├── hippocampus/          # Mémoire épisodique
├── memory/               # Mémoire sémantique + consolidation
├── thalamus/             # Gateway d'attention
├── prefrontal/           # Mémoire de travail
├── oscillations/         # Rythmes neuraux
└── attention/            # Contrôle attentionnel

scripts/
├── train_cognitive_v2.py # Entraînement avec toutes les améliorations ✨
├── train_wikipedia.py    # Entraînement original
├── test_cognitive_v2.py  # Tests des nouveaux composants
└── chat.py               # Interface de génération
```

---

## 🔬 Prochaines Étapes

- [ ] **Predictive Coding** : Implémentation complète de l'apprentissage prédictif
- [ ] **Cerebellum** : Module de timing et coordination motrice
- [ ] **Multi-modal** : Extension vers vision et audio
- [ ] **Meta-Learning** : Apprendre à apprendre (MAML-like biologique)

---

## 📖 Références Biologiques

- Hebb, D.O. (1949). *The Organization of Behavior*
- Schultz, W. (1998). Predictive reward signal of dopamine neurons. *J. Neurophysiology*
- Bienenstock, E., Cooper, L., Munro, P. (1982). Theory for the development of neuron selectivity (BCM rule)
- Buzsáki, G. (2006). *Rhythms of the Brain*

---

## 📜 Licence

MIT License - Voir [LICENSE](LICENSE)

---

<div align="center">

**Nōkai** - *L'IA qui pense comme un cerveau*

🧠 Made with neuroscience

</div>
