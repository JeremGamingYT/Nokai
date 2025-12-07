"""
Nōkai (脳海) - Bio-Inspired Artificial Brain Architecture

A revolutionary AI architecture that moves beyond statistical prediction
to achieve emergent understanding through biologically-inspired mechanisms.

Modules:
    - cortex: Cortical column processing units
    - hippocampus: Episodic memory and pattern completion
    - thalamus: Attention routing and gating
    - cerebellum: Temporal pattern learning
    - oscillations: Neural synchronization
    - learning: Hebbian plasticity rules
    - memory: External memory management
    - generation: Language generation pipeline
"""

__version__ = "0.1.0"
__author__ = "Nōkai Research Team"

from nokai.config import NokaiConfig
from nokai.model import NokaiModel

__all__ = [
    "NokaiConfig",
    "NokaiModel",
    "__version__",
]
