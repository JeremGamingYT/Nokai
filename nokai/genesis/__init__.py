"""
GENESIS - Generative Event-driven Neural Efficient Sparse Intelligent System

Nokai V2.0 Revolutionary Architecture

This module contains the next-generation neuromorphic components:
- TernaryWeight: Native {-1, 0, +1} weights
- RichNeuronUnit: Dynamic neurons with internal state
- GenesisLearning: Local STDP + RPE learning
- OscillatoryBinder: Phase-based concept binding
- ModernHopfieldMemory: One-shot episodic memory
- StructuralPlasticity: Dynamic topology
"""

from nokai.genesis.ternary import TernaryLinear, TernaryConv2d, ternary_quantize
from nokai.genesis.rnu import RichNeuronUnit, RNULayer
from nokai.genesis.learning import GenesisLearning, STDPRule, RPEComputation
from nokai.genesis.binding import OscillatoryBinder, PhaseCoding
from nokai.genesis.memory import ModernHopfieldMemory, WorkingMemoryBuffer
from nokai.genesis.structural import StructuralPlasticity, SynapticPruner
from nokai.genesis.neuromodulation import GenesisLimbic, Neuromodulator

__all__ = [
    # Core
    "TernaryLinear",
    "TernaryConv2d", 
    "ternary_quantize",
    "RichNeuronUnit",
    "RNULayer",
    # Learning
    "GenesisLearning",
    "STDPRule",
    "RPEComputation",
    # Binding
    "OscillatoryBinder",
    "PhaseCoding",
    # Memory
    "ModernHopfieldMemory",
    "WorkingMemoryBuffer",
    # Structure
    "StructuralPlasticity",
    "SynapticPruner",
    # Neuromodulation
    "GenesisLimbic",
    "Neuromodulator",
]
