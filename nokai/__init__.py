"""
Nōkai (脳海) - Bio-Inspired Artificial Brain Architecture

A revolutionary AI architecture that moves beyond statistical prediction
to achieve emergent understanding through biologically-inspired mechanisms.

=============================================================================
NEUROMORPHIC AGI ARCHITECTURE
=============================================================================

Core Modules (Brain Regions):
    1. THALAMUS     - Sensory gateway, attention filtering
    2. CORTEX       - Hierarchical processing, cortical columns
    3. HIPPOCAMPUS  - Episodic memory, pattern completion
    4. PREFRONTAL   - Working memory, executive control
    5. SEMANTIC     - Long-term knowledge (neocortex)
    6. LIMBIC       - Dopamine, reward, motivation
       - STRIATUM   - Decision making, action selection
       - dACC       - Metacognition, uncertainty monitoring
    7. OSCILLATIONS - Neural synchronization (theta/gamma)
    8. ATTENTION    - Dynamic resource allocation
    9. CONSOLIDATION- Sleep, memory transfer

Key Bio-Inspired Features:
    - synaptic_weights  : Plasticité synaptique (LTP/LTD)
    - energy_check()    : Sparsité métabolique
    - dopamine_level    : Modulation de l'apprentissage
    - oscillations      : Coordination inter-modules

See brain.py for the unified NeuromorphicBrain class.
"""

__version__ = "0.2.0"
__author__ = "Nōkai Research Team"

# Core configuration and models
from nokai.config import NokaiConfig
from nokai.model import NokaiModel
from nokai.brain import NeuromorphicBrain, BrainState

# Brain region modules
from nokai.cortex import Cortex
from nokai.hippocampus import HippocampalMemory
from nokai.thalamus import ThalamusGateway, ThalamicAttention
from nokai.prefrontal import PrefrontalWorkingMemory
from nokai.oscillations import OscillatorNetwork

# Limbic system (emotion/reward)
from nokai.limbic import (
    DopamineCircuit,
    RewardPredictionError,
    StriatumSelector,
    MetacognitiveMonitor,
)

# Memory systems
from nokai.memory import (
    SemanticMemory,
    ConsolidationSystem,
    SynapticHomeostasis,
)

# Attention and resource allocation
from nokai.attention import (
    AttentionController,
    ResourceAllocation,
    AdaptiveCompute,
)

# Learning rules
from nokai.learning import HebbianPlasticity, STDPRule

__all__ = [
    # Version info
    "__version__",
    
    # Core
    "NokaiConfig",
    "NokaiModel",
    "NeuromorphicBrain",
    "BrainState",
    
    # Brain regions
    "Cortex",
    "HippocampalMemory",
    "ThalamusGateway",
    "ThalamicAttention",
    "PrefrontalWorkingMemory",
    "OscillatorNetwork",
    
    # Limbic system
    "DopamineCircuit",
    "RewardPredictionError",
    "StriatumSelector",
    "MetacognitiveMonitor",
    
    # Memory
    "SemanticMemory",
    "ConsolidationSystem",
    "SynapticHomeostasis",
    
    # Attention
    "AttentionController",
    "ResourceAllocation",
    "AdaptiveCompute",
    
    # Learning
    "HebbianPlasticity",
    "STDPRule",
]
