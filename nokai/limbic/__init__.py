"""
Limbic System - The "Emotional" Core of Nōkai

This module implements the neurochemical simulation (Dopamine, 
Reward Prediction Error) and decision-making circuits inspired
by the brain's limbic system.

Components:
    - dopamine: VTA/Dopamine reward circuit (V1 - legacy)
    - dopamine_v2: Homeostatic dopamine with RPE-based surprise (V2 - recommended)
    - striatum: Action selection based on cost/benefit
    - dacc: Metacognitive monitoring (uncertainty estimation)
    
The V2 DopamineCircuit:
    - Implements true Reward Prediction Error (surprise, not raw success)
    - Includes homeostatic adaptation (hedonic treadmill)
    - Prevents the "drugged brain" problem where DA stays at 1.0
"""

# Import V2 as the default (V1 available as *Legacy)
from nokai.limbic.dopamine_v2 import (
    DopamineCircuitV2 as DopamineCircuit,
    DopamineStateV2 as DopamineState,
    ValueNetwork,
    NoveltyDetector,
    HomeostaticBaseline,
)

# Legacy V1 (for backwards compatibility)
from nokai.limbic.dopamine import (
    DopamineCircuit as DopamineCircuitV1,
    RewardPredictionError,
    DopamineState as DopamineStateV1,
)

from nokai.limbic.striatum import StriatumSelector
from nokai.limbic.dacc import MetacognitiveMonitor

__all__ = [
    # V2 (default)
    "DopamineCircuit",
    "DopamineState",
    "ValueNetwork",
    "NoveltyDetector",
    "HomeostaticBaseline",
    
    # V1 (legacy)
    "DopamineCircuitV1",
    "DopamineStateV1",
    "RewardPredictionError", 
    
    # Other limbic components
    "StriatumSelector",
    "MetacognitiveMonitor",
]
