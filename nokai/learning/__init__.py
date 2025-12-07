"""
Learning Module - Bio-inspired Learning Rules

Implements Hebbian plasticity and predictive coding as
alternatives to pure backpropagation.

Components:
    - hebbian: Classic Hebbian/Oja's rule (V1)
    - hebbian_v2: BCM metaplasticity + dopamine gating (V2 - recommended)
    - predictive: Predictive coding layers

The V2 HebbianLearner:
    - Implements BCM sliding threshold for metaplasticity
    - Dopamine-gated learning (only learn from rewarding experiences)
    - IMMEDIATE in-forward-pass updates (true local learning)
    - STDP eligibility traces for timing-dependent plasticity
"""

# V1 (legacy)
from nokai.learning.hebbian import (
    HebbianPlasticity as HebbianPlasticityV1,
    STDPRule as STDPRuleV1,
    LocalLearningLayer,
)

# V2 (default/recommended)
from nokai.learning.hebbian_v2 import (
    HebbianLearnerV2 as HebbianPlasticity,
    HebbianLinear,
    CorticalHebbianIntegrator,
    HebbianConfig,
    BCMThreshold,
    STDPTrace as STDPRule,
)

from nokai.learning.predictive import PredictiveCodingLayer

__all__ = [
    # V2 (default)
    "HebbianPlasticity",
    "HebbianLinear",
    "CorticalHebbianIntegrator",
    "HebbianConfig",
    "BCMThreshold",
    "STDPRule",
    
    # V1 (legacy)
    "HebbianPlasticityV1",
    "STDPRuleV1",
    "LocalLearningLayer",
    
    # Predictive coding
    "PredictiveCodingLayer",
]
