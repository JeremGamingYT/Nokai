# Nōkai : Architecture Cérébrale Artificielle

## Vision

**Nōkai** (脳海 - "océan cérébral" en japonais) est une architecture d'IA bio-inspirée qui s'écarte radicalement du paradigme des Large Language Models (LLM). Au lieu de prédire le prochain token de manière statistique, Nōkai simule les mécanismes fondamentaux du cerveau biologique pour atteindre une compréhension émergente du langage.

---

## 1. Principes Fondamentaux

### 1.1 Pourquoi les LLMs sont limités

| Limitation LLM | Solution Nōkai |
|----------------|----------------|
| Prédiction statistique sans compréhension | Représentations dynamiques et contextualisation active |
| Apprentissage par époques sur données statiques | Apprentissage continu en temps réel (plasticité) |
| Coût computationnel O(n²) pour attention | Attention sparse hiérarchique O(n log n) |
| Mémoire limitée au contexte | Mémoire associative externe illimitée |
| Nécessite des milliards de paramètres | Efficacité via sparsité et modularité |

### 1.2 Inspirations Biologiques

Nōkai s'inspire de cinq mécanismes cérébraux clés :

1. **Colonnes Corticales** : Unités de traitement modulaires et spécialisées
2. **Plasticité Hebbienne** : "Neurons that fire together, wire together"
3. **Ondes Cérébrales** : Synchronisation rythmique pour coordination globale
4. **Hippocampe** : Mémoire épisodique et consolidation
5. **Prédiction Active** : Le cerveau prédit constamment et apprend des erreurs

---

## 2. Architecture Hiérarchique

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NŌKAI BRAIN                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   CORTEX    │  │ HIPPOCAMPUS │  │  THALAMUS   │  │ CEREBELLUM  │ │
│  │  (Traitement│  │  (Mémoire   │  │  (Routage   │  │ (Timing &   │ │
│  │   Cognitif) │  │  Épisodique)│  │  Attention) │  │  Patterns)  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                              │                                       │
│                    ┌─────────▼─────────┐                            │
│                    │   GLOBAL WORKSPACE │                            │
│                    │  (Intégration)     │                            │
│                    └─────────┬─────────┘                            │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                  │
│         ▼                    ▼                    ▼                  │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐          │
│  │   INPUT     │      │   OUTPUT    │      │   MEMORY    │          │
│  │  (Encodage) │      │ (Décodage)  │      │  (Stockage) │          │
│  └─────────────┘      └─────────────┘      └─────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Composants Principaux

### 3.1 Colonnes Corticales (CorticalColumn)

Inspirées des minicolonnes du néocortex, ces unités sont les briques de base du traitement.

**Caractéristiques :**
- 1000-10000 colonnes spécialisées
- Chaque colonne = 100-1000 neurones
- Spécialisation émergente (syntaxe, sémantique, contexte...)
- Communication latérale sparse

```python
class CorticalColumn:
    neurons: int = 256          # Neurones par colonne
    specialization: str         # Émerge via apprentissage
    lateral_connections: sparse # Connexions aux colonnes voisines
    top_down_input: tensor      # Prédictions des niveaux supérieurs
    bottom_up_input: tensor     # Données des niveaux inférieurs
```

### 3.2 Mémoire Hippocampique (HippocampalMemory)

Système de mémoire épisodique pour stockage et récupération rapide.

**Mécanismes :**
- **Pattern Separation** : Encodages orthogonaux pour éviter interférences
- **Pattern Completion** : Reconstruction à partir d'indices partiels
- **Replay** : Consolidation pendant les phases de "sommeil"

```python
class HippocampalMemory:
    dentate_gyrus: PatternSeparator    # Orthogonalisation
    ca3_region: AutoAssociator         # Récupération par indice
    ca1_region: OutputMapper            # Vers Global Workspace
    memory_bank: VectorDatabase         # Stockage externe (faiss/hnswlib)
```

### 3.3 Attention Thalamique (ThalamicGating)

Le thalamus biologique filtre et route l'information. Ici, il implémente une attention dynamique ultra-efficace.

**Innovation clé : Attention O(n log n)**
- Clustering dynamique des tokens
- Routage sparse vers colonnes pertinentes
- Modulation par oscillations (voir section 4)

```python
class ThalamicGating:
    routing_network: nn.Linear   # Décide où router
    gating_mechanism: nn.Sigmoid # Ouverture/fermeture des canaux
    attention_clusters: HNSW     # Indexation hiérarchique
```

### 3.4 Timing Cérébelleux (CerebellarTiming)

Le cervelet gère la précision temporelle et les patterns séquentiels.

**Fonctions :**
- Prédiction du timing des séquences
- Apprentissage des patterns rythmiques
- Correction d'erreurs temporelles

---

## 4. Mécanismes d'Apprentissage

### 4.1 Plasticité Hebbienne Différentielle

Au lieu du backpropagation classique, Nōkai utilise une règle d'apprentissage locale inspirée de la biologie.

```python
# Règle de plasticité
Δw = η * (pre_activation * post_activation) - λ * w  # STDP simplifié

# Avantages :
# - Apprentissage local (pas de backward pass global)
# - Parallélisable massivement
# - Apprentissage continu sans catastrophic forgetting
```

### 4.2 Oscillations Neuronales

Les ondes cérébrales synchronisent l'activité :

| Bande | Fréquence | Fonction dans Nōkai |
|-------|-----------|---------------------|
| Delta | 1-4 Hz | Consolidation mémoire |
| Theta | 4-8 Hz | Encodage séquentiel |
| Alpha | 8-12 Hz | Inhibition/filtrage |
| Beta | 12-30 Hz | Traitement actif |
| Gamma | 30-100 Hz | Binding perceptuel |

**Implémentation :**
```python
class NeuralOscillator:
    phase: float              # Phase actuelle
    frequency: float          # Fréquence de base
    coupling: sparse_matrix   # Couplage entre oscillateurs
    
    def step(self, dt):
        # Kuramoto model pour synchronisation
        self.phase += 2π * self.frequency * dt + coupling_term
```

### 4.3 Prédictive Coding

Le cerveau prédit constamment ses entrées futures. Les erreurs de prédiction sont le signal d'apprentissage.

```
┌─────────────────────────────────────────┐
│ Niveau N+1 (Prédictions)                │
│         │ Prédiction ↓    ↑ Erreur      │
│         ▼                 │             │
│ Niveau N ─────────────────┴─────────────│
│         │ Prédiction ↓    ↑ Erreur      │
│         ▼                 │             │
│ Niveau N-1 (Plus proche des données)    │
└─────────────────────────────────────────┘
```

---

## 5. Optimisations pour Hardware Limité

### 5.1 Quantification Dynamique

```python
# Quantification adaptative basée sur l'importance
if neuron.activation_variance > threshold:
    precision = 16  # Neurones importants
else:
    precision = 4   # Neurones stables
```

### 5.2 Sparsité Structurelle

- **Activation Sparse** : <5% des neurones actifs simultanément
- **Connexions Sparse** : Matrice à 99% de zéros
- **Gradient Sparse** : Mise à jour sélective

### 5.3 Memory-Mapped Files

```python
# Données sur SSD, streaming vers GPU
class MemoryMappedCorpus:
    def __init__(self, path):
        self.mmap = np.memmap(path, dtype='float16', mode='r')
        
    def get_batch(self, indices):
        # Zero-copy vers GPU via DMA
        return torch.from_file(self.mmap, indices)
```

### 5.4 Gradient Checkpointing Intelligent

```python
# Sauvegarde sélective pour réduire mémoire
@checkpoint_if(memory_pressure > 0.8)
def forward(x):
    ...
```

---

## 6. Pipeline de Données Efficace

### 6.1 Prétraitement Offline

```
Corpus Texte → Tokenization → Embeddings → Compression → Index HNSW
     │              │              │            │           │
   100GB         20GB           10GB         2GB       500MB
```

### 6.2 Streaming Intelligent

```python
class SmartDataLoader:
    """
    Charge les données à la demande depuis le disque.
    Utilise prefetching prédictif basé sur patterns d'accès.
    """
    prefetch_queue: AsyncQueue
    access_predictor: MarkovChain
    
    async def get_batch(self):
        # Prédiction du prochain batch
        next_likely = self.access_predictor.predict()
        self.prefetch_queue.add(next_likely)
        return await self.current_batch
```

---

## 7. Architecture du Réseau

### 7.1 Dimensions Recommandées

| Config | Colonnes | Neurones/Col | Params | VRAM |
|--------|----------|--------------|--------|------|
| Nano | 256 | 64 | ~4M | 200MB |
| Micro | 512 | 128 | ~17M | 500MB |
| Mini | 1024 | 256 | ~67M | 2GB |
| Base | 2048 | 512 | ~268M | 6GB |
| Large | 4096 | 1024 | ~1B | 16GB |

### 7.2 Modularité

```python
# Configuration flexible
config = NokaiConfig(
    num_columns=1024,
    neurons_per_column=256,
    memory_size="external",  # Utilise SSD
    attention_type="sparse_thalamic",
    learning_rule="hebbian_stdp",
    oscillation_enabled=True,
)
```

---

## 8. Comparaison Théorique

### 8.1 Efficacité d'Apprentissage

| Aspect | LLM Traditionnel | Nōkai |
|--------|------------------|-------|
| Données nécessaires | Billions tokens | Millions tokens |
| Époques | 1-3 | Continu |
| Généralisation | Mémorisation partielle | Abstraction émergente |
| Catastrophic Forgetting | Oui | Résistant |

### 8.2 Efficacité Computationnelle

| Métrique | Transformer | Nōkai |
|----------|-------------|-------|
| Complexité Attention | O(n²) | O(n log n) |
| Activations | Dense | Sparse (<5%) |
| Mémoire Contexte | Limitée | Illimitée (externe) |
| Parallélisme | Limité | Massif (local) |

---

## 9. Limitations et Réalisme

### Ce que Nōkai peut réalistement atteindre :
✅ Apprentissage plus efficace en données que les LLMs  
✅ Mémoire à long terme via stockage externe  
✅ Entraînement sur GPU modeste (6GB) pour modèles <300M params  
✅ Apprentissage continu sans catastrophic forgetting  
✅ Inférence rapide grâce à la sparsité  

### Ce qui reste un défi de recherche :
⚠️ Égaler la qualité générative des LLMs de 70B+ params  
⚠️ Entraînement "des milliers de fois plus rapide" (amélioration réaliste: 10-50x)  
⚠️ Compréhension "réelle" reste philosophiquement débattu  

---

## 10. Roadmap d'Implémentation

### Phase 1 : Fondations (Semaines 1-4)
- [ ] Implémentation CorticalColumn
- [ ] Système de mémoire hippocampique
- [ ] Attention thalamique sparse
- [ ] Infrastructure de données

### Phase 2 : Apprentissage (Semaines 5-8)
- [ ] Règle de plasticité Hebbienne
- [ ] Oscillations neuronales
- [ ] Predictive coding
- [ ] Tests unitaires

### Phase 3 : Intégration (Semaines 9-12)
- [ ] Global Workspace
- [ ] Pipeline de génération
- [ ] Benchmarks
- [ ] Optimisations VRAM

### Phase 4 : Validation (Semaines 13-16)
- [ ] Comparaison avec baselines
- [ ] Ajustements architecture
- [ ] Documentation
- [ ] Release v1.0

---

## Références Scientifiques

1. Hawkins, J. (2021). *A Thousand Brains: A New Theory of Intelligence*
2. Friston, K. (2010). *The free-energy principle: a unified brain theory?*
3. Rao, R. & Ballard, D. (1999). *Predictive coding in the visual cortex*
4. Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*
5. O'Reilly, R. (2006). *Biologically Based Computational Models of Cortex*
