"""
Memory Package - All Memory Systems

This package provides the complete memory architecture:
    - HippocampalMemory: Episodic memory (fast encoding, pattern completion)
    - SemanticMemory: Long-term knowledge (slow consolidation)
    - ConsolidationSystem: Sleep/transfer between memories
"""

from nokai.memory.semantic import SemanticMemory
from nokai.memory.consolidation import ConsolidationSystem, SynapticHomeostasis

# Re-export hippocampal memory from existing location
from nokai.hippocampus import HippocampalMemory

__all__ = [
    "HippocampalMemory",
    "SemanticMemory",
    "ConsolidationSystem",
    "SynapticHomeostasis",
]
