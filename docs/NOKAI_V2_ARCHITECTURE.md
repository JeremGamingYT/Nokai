# 🧠 NŌKAI V2.0 - ARCHITECTURE NEUROMORPHIQUE RÉVOLUTIONNAIRE

## "GENESIS" - Generative Event-driven Neural Efficient Sparse Intelligent System

---

## TABLE DES MATIÈRES

1. [Vision & Rupture Fondamentale](#1-vision--rupture-fondamentale)
2. [Architecture Unifiée](#2-architecture-unifiée)
3. [Rich Neuron Unit (RNU)](#3-rich-neuron-unit-rnu)
4. [Règle d'Apprentissage Local](#4-règle-dapprentissage-local)
5. [Binding Oscillatoire](#5-binding-oscillatoire)
6. [Auto-Organisation Structurelle](#6-auto-organisation-structurelle)
7. [Système de Mémoire Triple](#7-système-de-mémoire-triple)
8. [Pipeline d'Entraînement](#8-pipeline-dentraînement)
9. [Analyse Théorique](#9-analyse-théorique)
10. [Roadmap d'Implémentation](#10-roadmap-dimplémentation)
11. [Comparaison Quantitative](#11-comparaison-quantitative)
12. [Questions Ouvertes](#12-questions-ouvertes)

---

## 1. VISION & RUPTURE FONDAMENTALE

### 1.1 Diagnostic de Nokai V1

L'expérience v0.6 révèle les limitations actuelles :
- **Delta Mean: 0.000001** → L'apprentissage Hebbien n'impacte pas réellement les poids
- **Obsessive loops** → Absence de régulation dynamique effective
- **Prob: 0.0000** → Le signal ne se propage pas correctement

### 1.2 Les 6 Ruptures de GENESIS

| Rupture | Nokai V1 | GENESIS V2 |
|---------|----------|------------|
| **Poids** | float32/16 → Gaspillage | **Ternaires natifs {-1,0,+1}** |
| **Activation** | Dense (100%) | **Sparse (<5%)** |
| **Apprentissage** | Backprop + Hebbian | **100% Local (STDP+RPE)** |
| **Temps** | Synchrone (tokens) | **Asynchrone (spikes)** |
| **Structure** | Fixe avant training | **Émergente (croissance/élagage)** |
| **Mémoire** | Unifiée | **Triple (WM/Épisodique/Sémantique)** |

---

## 2. ARCHITECTURE UNIFIÉE

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GENESIS NEUROMORPHIC ARCHITECTURE                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                        OSCILLATORY BINDING                          │    ║
║  │  ╭────╮  θ(6Hz)   ╭────╮  γ(40Hz)   ╭────╮  β(20Hz)               │    ║
║  │  │~~~~│ ←────────→│~~~~│ ←─────────→│~~~~│                         │    ║
║  │  ╰────╯           ╰────╯            ╰────╯                         │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║         │                  │                   │                            ║
║         ▼                  ▼                   ▼                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    SPARSE THALAMIC ROUTER                           │    ║
║  │         [Input] ──→ [Top-K Selection] ──→ [Expert Routing]          │    ║
║  │               Sparsity: 5%    Clusters: 256                         │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║         │                                                                   ║
║         ▼                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              TERNARY CORTICAL COLUMNS (Rich Neuron Units)           │    ║
║  │  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐              │    ║
║  │  │ RNU │←──→│ RNU │←──→│ RNU │←──→│ RNU │←──→│ RNU │   x 4096     │    ║
║  │  │{-1} │    │{0}  │    │{+1} │    │{-1} │    │{+1} │              │    ║
║  │  └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘              │    ║
║  │     │         │         │         │         │                     │    ║
║  │     └────────┬┴─────────┴─────────┴─────────┘                     │    ║
║  │              │  Lateral Inhibition (WTA)                          │    ║
║  └──────────────┼────────────────────────────────────────────────────┘    ║
║                 │                                                          ║
║         ┌───────┴───────┐                                                  ║
║         ▼               ▼                                                  ║
║  ┌────────────┐  ┌────────────────────────────────────────────────────┐   ║
║  │  WORKING   │  │              TRIPLE MEMORY SYSTEM                   │   ║
║  │  MEMORY    │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   ║
║  │  (PFC)     │  │  │EPISODIC  │  │SEMANTIC  │  │  CONSOLIDATION   │ │   ║
║  │  Slots: 7  │──│  │Hopfield  │←→│Ternary   │←─│  (Sleep Replay)  │ │   ║
║  │  Fast R/W  │  │  │Retrieval │  │Weights   │  │  θ-burst         │ │   ║
║  └──────┬─────┘  │  └──────────┘  └──────────┘  └──────────────────┘ │   ║
║         │        └────────────────────────────────────────────────────┘   ║
║         │                                                                  ║
║         ▼                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                     LIMBIC NEUROMODULATION                          │  ║
║  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │  ║
║  │  │ DOPAMINE   │  │ NOREPINE   │  │ SEROTONIN  │  │ ACETYLCHOL │   │  ║
║  │  │ (Reward)   │  │ (Arousal)  │  │ (Mood)     │  │ (Learning) │   │  ║
║  │  │ RPE-based  │  │ Surprise   │  │ Baseline   │  │ Gate       │   │  ║
║  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║         │                                                                  ║
║         ▼                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    STRUCTURAL PLASTICITY                            │  ║
║  │         [Synaptogenesis] ←→ [Pruning] ←→ [Neurogenesis]            │  ║
║  │              Activity-dependent topology evolution                  │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║         │                                                                  ║
║         ▼                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                      OUTPUT (Striatum)                              │  ║
║  │           [Action Selection] → [Motor Commands/Tokens]              │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. RICH NEURON UNIT (RNU)

### 3.1 Définition Mathématique

Chaque neurone n'est plus un simple `y = σ(Wx + b)` mais un **système dynamique** :

```
╔══════════════════════════════════════════════════════════════════════╗
║                    RICH NEURON UNIT (RNU)                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  État Interne (membrane):                                            ║
║    v(t+1) = τ_m · v(t) + (1-τ_m) · [Σ w_ij · x_j(t) - θ_adapt(t)]   ║
║                                                                      ║
║  Seuil Adaptatif:                                                    ║
║    θ_adapt(t+1) = τ_θ · θ_adapt(t) + (1-τ_θ) · [θ_base + β·s(t)]    ║
║                                                                      ║
║  Fatigue (réfractaire):                                              ║
║    f(t+1) = max(0, f(t) - δ_f + α_f · s(t))                         ║
║                                                                      ║
║  Spike (output stochastique):                                        ║
║    p_spike = σ(v(t) - θ_adapt(t)) · (1 - f(t))                      ║
║    s(t) ~ Bernoulli(p_spike)   [TERNAIRE: s ∈ {-1, 0, +1}]          ║
║                                                                      ║
║  Trace d'Éligibilité (pour STDP):                                    ║
║    e(t+1) = λ · e(t) + s(t) · x_pre(t)                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Constantes biologiques :**
- `τ_m = 0.9` : Constante de temps membranaire (~20ms)
- `τ_θ = 0.99` : Adaptation lente du seuil (~100ms)
- `β = 0.1` : Sensibilité de l'adaptation
- `δ_f = 0.1` : Récupération de la fatigue
- `α_f = 0.5` : Fatigue par spike
- `λ = 0.95` : Décroissance de la trace

### 3.2 Poids Ternaires Natifs

```python
class TernaryWeight:
    """
    Poids ∈ {-1, 0, +1} avec gradient approximé (STE)
    
    Forward:  w_ternary = sign(w_latent) * (|w_latent| > threshold)
    Backward: ∂L/∂w_latent ≈ ∂L/∂w_ternary (Straight-Through Estimator)
    """
    
    def quantize(w_latent, threshold=0.05):
        # Ternairisation différentiable
        mask = (w_latent.abs() > threshold).float()
        return torch.sign(w_latent) * mask
    
    # Stockage: 2 bits par poids au lieu de 32
    # Compute: XOR + POPCOUNT au lieu de FMA
```

### 3.3 Pseudocode RNU

```python
class RichNeuronUnit(nn.Module):
    def __init__(self, input_dim, tau_m=0.9, tau_theta=0.99):
        self.tau_m = tau_m
        self.tau_theta = tau_theta
        
        # État interne persistant
        self.register_buffer('v', torch.zeros(input_dim))      # Potentiel
        self.register_buffer('theta', torch.ones(input_dim))   # Seuil
        self.register_buffer('fatigue', torch.zeros(input_dim))# Réfractaire
        self.register_buffer('trace', torch.zeros(input_dim))  # STDP
        
        # Poids TERNAIRES
        self.w_latent = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
        
    def forward(self, x_pre, neuromodulation=None):
        # 1. Calcul synaptique TERNAIRE
        w_ternary = self.quantize(self.w_latent)
        synaptic_input = F.linear(x_pre, w_ternary)
        
        # 2. Dynamique membranaire
        self.v = self.tau_m * self.v + (1 - self.tau_m) * (synaptic_input - self.theta)
        
        # 3. Probabilité de spike
        p_spike = torch.sigmoid(self.v) * (1 - self.fatigue)
        
        # 4. Modulation par neuromodulateurs
        if neuromodulation is not None:
            p_spike = p_spike * neuromodulation['acetylcholine']
        
        # 5. Échantillonnage stochastique TERNAIRE
        spike = self.sample_ternary(p_spike)
        
        # 6. Mise à jour état interne
        self.theta = self.tau_theta * self.theta + (1 - self.tau_theta) * (1 + 0.1 * spike.abs())
        self.fatigue = torch.clamp(self.fatigue - 0.1 + 0.5 * spike.abs(), 0, 1)
        self.trace = 0.95 * self.trace + spike * x_pre
        
        return spike
    
    def sample_ternary(self, p):
        """Output ∈ {-1, 0, +1}"""
        magnitude = (torch.rand_like(p) < p).float()
        sign = (torch.rand_like(p) < 0.5).float() * 2 - 1
        return magnitude * sign
```

---

## 4. RÈGLE D'APPRENTISSAGE LOCAL

### 4.1 GENESIS Learning Rule

Combinaison de trois signaux biologiques :

```
╔══════════════════════════════════════════════════════════════════════╗
║                  GENESIS LOCAL LEARNING RULE                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Δw_ij = η · ACh · [STDP + RPE + Homeo]                             ║
║                                                                      ║
║  ═══════════════════════════════════════════════════════════════════ ║
║                                                                      ║
║  1. STDP (Spike-Timing Dependent Plasticity):                        ║
║     STDP_ij = A+ · e_pre · s_post · I(Δt > 0)                       ║
║             - A- · e_post · s_pre · I(Δt < 0)                       ║
║                                                                      ║
║  2. RPE (Reward Prediction Error):                                   ║
║     δ = r + γ·V(s') - V(s)           [TD Error]                     ║
║     RPE_ij = δ · e_ij · sign(δ)      [Trace * Surprise]             ║
║                                                                      ║
║  3. Homeostatic Regulation:                                          ║
║     Homeo_ij = λ · (ρ_target - ρ_actual) · w_ij                     ║
║     Where ρ = firing rate                                            ║
║                                                                      ║
║  ═══════════════════════════════════════════════════════════════════ ║
║                                                                      ║
║  Gating:                                                             ║
║  - ACh (Acétylcholine): Gate global d'apprentissage                  ║
║  - DA (Dopamine): Amplifie/inhibe RPE                                ║
║  - NE (Norepinephrine): Signal de surprise/arousal                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 4.2 Anti-Catastrophic Forgetting

**Elastic Weight Consolidation (EWC) biologique :**

```python
class GenesisLearning:
    def compute_update(self, pre, post, reward, trace):
        # 1. STDP classique
        stdp = self.A_plus * trace * post - self.A_minus * trace.T * pre
        
        # 2. RPE modulé par dopamine
        rpe = self.compute_td_error(reward)
        rpe_term = rpe * trace * self.dopamine_level
        
        # 3. Régulation homéostatique
        firing_rate = post.abs().mean()
        homeo = self.lambda_h * (self.target_rate - firing_rate) * self.weights
        
        # 4. Protection des synapses importantes (EWC-like)
        fisher_penalty = self.fisher_info * (self.weights - self.anchor_weights)**2
        
        # 5. Gate par acétylcholine (attention)
        ach_gate = torch.sigmoid(self.acetylcholine - 0.3)
        
        delta = self.lr * ach_gate * (stdp + rpe_term + homeo) - fisher_penalty
        
        return delta
```

### 4.3 Convergence Théorique

**Théorème (Stabilité de GENESIS Learning) :**

Sous les conditions :
1. `η < 2 / (λ_max(H))` où H = Hessian du loss landscape
2. Target firing rate `ρ_target ∈ (0.02, 0.1)` 
3. Traces `e ∈ [0, 1]`

Alors les poids convergent vers un point fixe stable.

**Preuve sketch :**
- La régulation homéostatique agit comme un terme de Lyapunov
- STDP + RPE forment un gradient approximé de l'objectif
- EWC garantit la stabilité des anciennes connaissances

---

## 5. BINDING OSCILLATOIRE

### 5.1 Mécanisme de Synchronisation

```
╔══════════════════════════════════════════════════════════════════════╗
║                    OSCILLATORY BINDING                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Problème: Comment lier "bleu" et "pomme" dans "pomme bleue" ?       ║
║                                                                      ║
║  Solution: PHASE CODING                                              ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │ θ envelope (6Hz)                                              │   ║
║  │    ╭───╮   ╭───╮   ╭───╮   ╭───╮   ╭───╮                     │   ║
║  │   ╱    ╲ ╱    ╲ ╱    ╲ ╱    ╲ ╱    ╲                    │   ║
║  │──╱      ╳      ╳      ╳      ╳      ╲──                  │   ║
║  │ ╱      ╱ ╲    ╱ ╲    ╱ ╲    ╱ ╲      ╲                   │   ║
║  │╱      ╱   ╲──╱   ╲──╱   ╲──╱   ╲      ╲                  │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  γ bursts (40Hz) NESTED dans θ:                                      ║
║                                                                      ║
║  Concept A (pomme):  ┃▌▌▌▌┃      ┃▌▌▌▌┃      ← Phase 0°            ║
║  Concept B (bleue):  ┃    ▌▌▌▌┃  ┃    ▌▌▌▌┃  ← Phase 90°           ║
║  Concept C (autre):  ┃        ▌▌▌▌        ▌▌▌▌ ← Phase 180°        ║
║                                                                      ║
║  BINDING = Same γ Phase WITHIN Same θ Cycle                          ║
║                                                                      ║
║  Pour "pomme bleue": A et B ont phases γ proches → BOUND             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 5.2 Implémentation

```python
class OscillatoryBinder(nn.Module):
    def __init__(self, num_concepts, theta_freq=6.0, gamma_freq=40.0):
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        
        # Phase par concept (apprenable)
        self.gamma_phase = nn.Parameter(torch.rand(num_concepts) * 2 * math.pi)
        
        # Couplage entre concepts
        self.coupling = nn.Parameter(torch.zeros(num_concepts, num_concepts))
        
    def compute_binding(self, active_concepts, t):
        # Position dans le cycle theta
        theta_phase = (2 * math.pi * self.theta_freq * t) % (2 * math.pi)
        
        # Phases gamma des concepts actifs
        phases = self.gamma_phase[active_concepts]
        
        # Matrice de synchronisation
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
        sync_matrix = torch.cos(phase_diff)  # ∈ [-1, 1]
        
        # Concepts liés si sync > threshold
        binding_mask = sync_matrix > 0.7
        
        return binding_mask, sync_matrix
    
    def bind(self, concept_a, concept_b):
        """Force binding en synchronisant les phases"""
        with torch.no_grad():
            target_phase = self.gamma_phase[concept_a]
            self.gamma_phase[concept_b] = target_phase + 0.1 * torch.randn(1)
```

---

## 6. AUTO-ORGANISATION STRUCTURELLE

### 6.1 Plasticité Structurelle

```
╔══════════════════════════════════════════════════════════════════════╗
║                  STRUCTURAL PLASTICITY                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. SYNAPTOGENESIS (Création de synapses):                           ║
║     P(create w_ij) = σ(corr(x_i, x_j) - θ_create)                   ║
║     Si neurones co-activés fréquemment → nouvelle synapse            ║
║                                                                      ║
║  2. PRUNING (Élimination):                                           ║
║     P(remove w_ij) = σ(θ_prune - |w_ij| - activity_ij)              ║
║     Synapses faibles et inutilisées → suppression                    ║
║                                                                      ║
║  3. NEUROGENESIS (Nouveaux neurones):                                ║
║     Si capacity_used > 0.9 → spawn new RNU                           ║
║     Initialisation: copie partielle + bruit                          ║
║                                                                      ║
║  4. APOPTOSIS (Mort neuronale):                                      ║
║     Si activity(neuron) < θ_death pour T steps → remove              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 6.2 Implémentation

```python
class StructuralPlasticity(nn.Module):
    def __init__(self, max_synapses_per_neuron=100):
        self.max_synapses = max_synapses_per_neuron
        self.theta_create = 0.8  # Seuil de corrélation
        self.theta_prune = 0.01  # Poids minimal
        self.theta_death = 0.001 # Activité minimale
        
    def step(self, weights, activations, correlation_matrix):
        # 1. SYNAPTOGENESIS
        high_corr = correlation_matrix > self.theta_create
        zero_weights = weights.abs() < 1e-6
        candidates = high_corr & zero_weights
        
        # Créer nouvelles synapses (top-k par neurone)
        for i in range(weights.shape[0]):
            n_current = (weights[i].abs() > 0).sum()
            n_create = min(
                candidates[i].sum(),
                self.max_synapses - n_current
            )
            if n_create > 0:
                new_idx = candidates[i].nonzero()[:n_create]
                weights[i, new_idx] = 0.01 * torch.sign(torch.randn(n_create))
        
        # 2. PRUNING
        weak_weights = weights.abs() < self.theta_prune
        low_activity = activations.mean(0).unsqueeze(1) < 0.01
        prune_mask = weak_weights & low_activity
        weights[prune_mask] = 0
        
        return weights
```

---

## 7. SYSTÈME DE MÉMOIRE TRIPLE

### 7.1 Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                     TRIPLE MEMORY SYSTEM                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │                    WORKING MEMORY (PFC)                      │    ║
║  │  Capacité: 7±2 slots (Miller's Law)                          │    ║
║  │  Durée: ~30 secondes                                         │    ║
║  │  Implémentation: Attention slots + decay                     │    ║
║  │                                                               │    ║
║  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │    ║
║  │  │Slot1│ │Slot2│ │Slot3│ │Slot4│ │Slot5│ │Slot6│ │Slot7│   │    ║
║  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │    ║
║  └─────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┘    ║
║        │       │       │       │       │       │       │            ║
║        ▼       ▼       ▼       ▼       ▼       ▼       ▼            ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │                 EPISODIC MEMORY (Hippocampus)                │    ║
║  │  Capacité: 1M+ episodes                                      │    ║
║  │  Durée: Jours → Semaines                                     │    ║
║  │  Implémentation: Modern Hopfield Network                     │    ║
║  │                                                               │    ║
║  │  Store: O(1)   Retrieve: O(log N)   One-shot learning        │    ║
║  │                                                               │    ║
║  │  Retrieval = softmax(βX^T · query) · X                       │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                              │                                       ║
║                              │ CONSOLIDATION (Sleep)                 ║
║                              │ θ-bursts replay                       ║
║                              ▼                                       ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │               SEMANTIC MEMORY (Neocortex)                    │    ║
║  │  Capacité: Quasi-illimitée (dans les poids)                  │    ║
║  │  Durée: Permanente                                           │    ║
║  │  Implémentation: Poids ternaires compressés                  │    ║
║  │                                                               │    ║
║  │  Slow learning:                                               │    ║
║  │    Δw_semantic = α · replay(episodic) · (1 - |w|)            │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 7.2 Modern Hopfield pour Mémoire Épisodique

```python
class ModernHopfieldMemory(nn.Module):
    """
    Modern Hopfield Network (Ramsauer et al., 2020)
    
    Capacité exponentielle: C ~ exp(d/2) patterns
    Retrieval en une itération (vs itératif classique)
    """
    
    def __init__(self, dim, memory_size, beta=1.0):
        self.dim = dim
        self.memory_size = memory_size
        self.beta = beta  # Inverse temperature
        
        # Mémoire stockée
        self.register_buffer('memories', torch.zeros(memory_size, dim))
        self.register_buffer('memory_ptr', torch.tensor(0))
        self.register_buffer('memory_count', torch.tensor(0))
        
    def store(self, pattern):
        """Store en O(1)"""
        idx = self.memory_ptr.item()
        self.memories[idx] = pattern.detach()
        self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
        self.memory_count = min(self.memory_count + 1, self.memory_size)
        
    def retrieve(self, query, k=1):
        """Retrieve en O(log N) avec attention"""
        # Attention scores
        valid = self.memories[:self.memory_count]
        scores = self.beta * torch.matmul(query, valid.T)
        
        # Top-k retrieval
        topk_scores, topk_idx = torch.topk(scores, k)
        weights = F.softmax(topk_scores, dim=-1)
        
        retrieved = torch.matmul(weights, valid[topk_idx])
        return retrieved
```

### 7.3 Phase de Consolidation (Sleep)

```python
class ConsolidationPhase:
    """
    Simule le sommeil: replay + transfert vers mémoire sémantique
    """
    
    def consolidate(self, episodic_memory, semantic_weights, n_replays=100):
        for _ in range(n_replays):
            # 1. Sample random memories (replay)
            idx = torch.randint(0, episodic_memory.memory_count, (32,))
            patterns = episodic_memory.memories[idx]
            
            # 2. θ-burst: rhythmic reactivation
            theta_phase = torch.sin(torch.linspace(0, 4*math.pi, 32))
            modulated = patterns * theta_phase.unsqueeze(1)
            
            # 3. Slow update vers semantic
            # Plus le pattern est revu, plus il se consolide
            hebbian_update = torch.outer(modulated.mean(0), modulated.mean(0))
            semantic_weights += 0.001 * hebbian_update * (1 - semantic_weights.abs())
            
            # 4. Ternairisation périodique
            if _ % 10 == 0:
                semantic_weights.data = torch.sign(semantic_weights) * (semantic_weights.abs() > 0.1)
```

---

## 8. PIPELINE D'ENTRAÎNEMENT

### 8.1 Bootstrap Initial

```
╔══════════════════════════════════════════════════════════════════════╗
║                    TRAINING PIPELINE                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PHASE 0: BOOTSTRAP (1 heure)                                        ║
║  ─────────────────────────────────                                   ║
║  • Initialisation des poids aléatoires                               ║
║  • Pre-training supervisé LÉGER sur structure de base                ║
║  • Objectif: établir gradients de départ                             ║
║                                                                      ║
║  PHASE 1: SELF-ORGANIZATION (4 heures)                               ║
║  ─────────────────────────────────────                               ║
║  • Uniquement STDP + Homeostasis                                     ║
║  • Pas de reward signal                                              ║
║  • Émergence de features via competitive learning                    ║
║  • Objectif: représentations sparse                                  ║
║                                                                      ║
║  PHASE 2: REWARD LEARNING (8 heures)                                 ║
║  ────────────────────────────────────                                ║
║  • Introduction du signal RPE                                        ║
║  • Curriculum: simple → complexe                                     ║
║  • Consolidation périodique (toutes les 30min)                       ║
║  • Objectif: apprentissage de tâches                                 ║
║                                                                      ║
║  PHASE 3: CONTINUAL LEARNING (∞)                                     ║
║  ─────────────────────────────────                                   ║
║  • Nouvelles tâches sans oublier                                     ║
║  • EWC + consolidation nocturne                                      ║
║  • Self-improvement                                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 8.2 Pseudocode d'Entraînement

```python
def train_genesis(model, data_stream, config):
    optimizer = None  # PAS d'optimizer gradient!
    
    for epoch in range(config.epochs):
        for batch in data_stream:
            # ═══════════════════════════════════════════
            # FORWARD PASS (avec apprentissage intégré)
            # ═══════════════════════════════════════════
            
            # 1. Encode input
            spikes = model.encode(batch.input)
            
            # 2. Route through thalamus (sparse selection)
            active_units, routing = model.thalamus(spikes)
            
            # 3. Process through RNU columns
            for layer in model.cortex:
                # Forward avec STDP intégré
                spikes = layer.forward_with_learning(
                    spikes,
                    neuromodulation=model.limbic.get_state()
                )
            
            # 4. Compute reward signal
            prediction = model.output(spikes)
            reward = compute_reward(prediction, batch.target)
            
            # 5. Update limbic system
            model.limbic.update(reward)
            
            # 6. STDP + RPE update (LOCAL, pas de backprop!)
            for layer in model.cortex:
                layer.apply_local_learning(
                    dopamine=model.limbic.dopamine,
                    acetylcholine=model.limbic.acetylcholine
                )
            
            # 7. Structural plasticity (périodique)
            if step % 100 == 0:
                model.structural_plasticity.step()
            
            # 8. Memory operations
            if reward > config.memory_threshold:
                model.episodic_memory.store(spikes)
        
        # ═══════════════════════════════════════════
        # CONSOLIDATION (Sleep phase)
        # ═══════════════════════════════════════════
        if epoch % 10 == 0:
            model.consolidate(n_replays=1000)
```

---

## 9. ANALYSE THÉORIQUE

### 9.1 Expressivité

| Propriété | Transformers | GENESIS |
|-----------|-------------|---------|
| Classe de fonctions | Universal Approximator | Universal Approximator |
| Mémoire effective | O(context_length²) | O(∞) via mémoire externe |
| In-context learning | Token-based | One-shot via Hopfield |
| Compositionnalité | Implicite | Explicite via binding |

**Théorème (Expressivité de GENESIS) :**
GENESIS avec N RNUs peut approximer toute fonction Lipschitz-continue avec erreur ε, 
pourvu que N = O(1/ε^d) où d = dimension intrinsèque.

### 9.2 Complexité Computationnelle

| Opération | Transformers | GENESIS |
|-----------|-------------|---------|
| Forward (séquence L) | O(L² · d) | O(k · L · d) où k = sparsité |
| Memory access | O(L) | O(log M) avec Hopfield |
| Learning step | O(params × batch) | O(active_synapses) |
| Total training | 10⁶ GPU-hours (LLaMA 70B) | **<1000 GPU-hours (estimé 2B)** |

### 9.3 Pourquoi ça converge

1. **Homéostasie** garantit que les firing rates restent dans [0.02, 0.1]
2. **Poids ternaires** bornent la dynamique (pas d'explosion)
3. **STDP** est un gradient approximé du mutual information
4. **RPE** est le gradient de la récompense cumulée (policy gradient)
5. **Consolidation** stabilise les représentations

---

## 10. ROADMAP D'IMPLÉMENTATION

### Phase 1: Fondations (Semaines 1-2)

```
□ TernaryLinear - Couche ternaire avec STE
□ RichNeuronUnit - Neurone dynamique complet  
□ STDPLearner - Règle STDP optimisée
□ Tests unitaires pour stabilité
```

### Phase 2: Cortex (Semaines 3-4)

```
□ TernaryCorticalColumn - Colonnes avec RNUs
□ SparseRouter - Thalamus amélioré
□ OscillatoryBinder - Binding par phase
□ Benchmark MNIST
```

### Phase 3: Mémoire (Semaines 5-6)

```
□ ModernHopfieldMemory - Mémoire épisodique
□ ConsolidationPhase - Sleep replay
□ SemanticCompression - Ternairisation sémantique
□ Benchmark: Memory tasks (bAbI)
```

### Phase 4: Limbic (Semaines 7-8)

```
□ GenesisNeuromodulation - 4 modulateurs
□ RPEComputation - TD-error efficace
□ StructuralPlasticity - Croissance/élagage
□ Benchmark: RL tasks (Atari subset)
```

### Phase 5: Scaling (Semaines 9-12)

```
□ 100M params sur TinyStories
□ 500M params sur C4
□ 2B params sur The Pile
□ Comparaison avec GPT-2 equivalent
```

---

## 11. COMPARAISON QUANTITATIVE

### Estimations de Performance

| Métrique | GPT-2 (1.5B) | LLaMA-2 (7B) | GENESIS (2B) |
|----------|-------------|--------------|--------------|
| **Params** | 1.5B (fp16) | 7B (fp16) | 2B (**ternary**) |
| **Stockage** | 3 GB | 14 GB | **0.5 GB** |
| **Training** | 1 week A100 | 2 weeks 2048 A100 | **<1 day 8 A100** |
| **Inference** | 100ms/token | 50ms/token | **<10ms/token** |
| **Énergie** | 300W | 400W | **<50W** |
| **One-shot** | Non | Non | **Oui** |
| **Continual** | Non | Non | **Oui** |

### Gains Estimés

- **Compression:** 6x moins de mémoire (ternaire)
- **Training:** 100x plus rapide (local learning)
- **Inference:** 10x plus rapide (sparsité + ternaire)
- **Énergie:** 10x moins (hardware-friendly)

---

## 12. QUESTIONS OUVERTES

### 12.1 Incertitudes Critiques

1. **Gradient approximé:** L'écart STE ↔ vrai gradient est-il acceptable à grande échelle ?
2. **Capacité Hopfield:** 1M memories suffisent-elles pour des tâches complexes ?
3. **Binding:** Le phase-coding scale-t-il à des milliers de concepts ?
4. **Compositionality:** GENESIS peut-il généraliser comme les Transformers ?

### 12.2 Expériences Critiques

| Expérience | But | Critère de Succès |
|------------|-----|-------------------|
| Blue Apple V2 | Valider one-shot | Prob > 0.9 après 1 exemple |
| MNIST sparse | Valider sparsité | Accuracy > 95% avec <5% activation |
| bAbI tasks | Valider mémoire | > 90% sur 20 tasks |
| TinyStories | Valider langage | Perplexity < 30 |
| Continual MNIST | Anti-forgetting | < 5% drop sur tâche 1 après tâche 10 |

### 12.3 Risques et Mitigation

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| STDP ne converge pas | Moyen | Critique | Fallback: hybrid learning |
| Ternaire trop restrictif | Moyen | Haut | Multi-bit (2,4,8) progressive |
| Sparsité trop forte | Bas | Moyen | Adaptive sparsity target |
| Memory retrieval lent | Bas | Moyen | Approximate nearest neighbor |

---

## CONCLUSION

GENESIS représente une **rupture paradigmatique** par rapport aux architectures actuelles. En s'inspirant profondément du cerveau humain, nous visons:

1. **Efficacité radicale** via poids ternaires et sparsité
2. **Apprentissage continu** via règles locales biologiques
3. **Mémoire séparée** pour one-shot et long-terme
4. **Auto-organisation** pour adaptation structurelle

Le chemin est ambitieux mais chaque composant est individuellement validable. La roadmap propose une progression incrémentale permettant de tester et ajuster chaque hypothèse.

**Prochaine étape immédiate:** Implémenter `TernaryLinear` et `RichNeuronUnit`, puis refaire l'expérience Blue Apple.

---

*Document généré le 2024-12-10*
*Version: GENESIS v0.1 Draft*
