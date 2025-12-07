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

NEW in 0.3.0:
    - BPE Tokenization  : Semantic understanding (not char-level)
    - Homeostatic DA    : Dopamine adapts to prevent saturation
    - Immediate Hebbian : Local learning during forward pass

See brain.py for the unified NeuromorphicBrain class.
"""

__version__ = "0.3.0"
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

# Limbic system (emotion/reward) - V2 with homeostasis
from nokai.limbic import (
    DopamineCircuit,  # V2 with homeostasis (default)
    DopamineState,
    DopamineCircuitV1,  # Legacy
    RewardPredictionError,
    StriatumSelector,
    MetacognitiveMonitor,
    HomeostaticBaseline,
    NoveltyDetector,
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

# Learning rules - V2 with BCM metaplasticity
from nokai.learning import (
    HebbianPlasticity,  # V2 with BCM (default)
    HebbianLinear,
    CorticalHebbianIntegrator,
    HebbianConfig,
    STDPRule,
    PredictiveCodingLayer,
)

# Tokenization - BPE for semantic understanding
try:
    from nokai.tokenization import (
        NokaiTokenizer,
        SimpleBPETokenizer,
        TokenizerConfig,
        create_tokenizer,
        HAS_TOKENIZERS,
    )
    _HAS_TOKENIZATION = True
except ImportError:
    _HAS_TOKENIZATION = False
    HAS_TOKENIZERS = False

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
    
    # Limbic system (V2 default)
    "DopamineCircuit",
    "DopamineState",
    "DopamineCircuitV1",
    "RewardPredictionError",
    "StriatumSelector",
    "MetacognitiveMonitor",
    "HomeostaticBaseline",
    "NoveltyDetector",
    
    # Memory
    "SemanticMemory",
    "ConsolidationSystem",
    "SynapticHomeostasis",
    
    # Attention
    "AttentionController",
    "ResourceAllocation",
    "AdaptiveCompute",
    
    # Learning (V2 default)
    "HebbianPlasticity",
    "HebbianLinear",
    "CorticalHebbianIntegrator",
    "HebbianConfig",
    "STDPRule",
    "PredictiveCodingLayer",
    
    # Tokenization
    "NokaiTokenizer",
    "SimpleBPETokenizer",
    "TokenizerConfig",
    "create_tokenizer",
    "HAS_TOKENIZERS",
]
