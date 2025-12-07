"""
Limbic System - The "Emotional" Core of Nōkai

This module implements the neurochemical simulation (Dopamine, 
Reward Prediction Error) and decision-making circuits inspired
by the brain's limbic system.

Components:
    - dopamine: VTA/Dopamine reward circuit
    - striatum: Action selection based on cost/benefit
    - dacc: Metacognitive monitoring (uncertainty estimation)
    - reward: Reward Prediction Error (RPE) computation
"""

from nokai.limbic.dopamine import DopamineCircuit, RewardPredictionError
from nokai.limbic.striatum import StriatumSelector
from nokai.limbic.dacc import MetacognitiveMonitor

__all__ = [
    "DopamineCircuit",
    "RewardPredictionError", 
    "StriatumSelector",
    "MetacognitiveMonitor",
]
