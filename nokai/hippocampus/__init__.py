"""
Hippocampus Module - Episodic Memory System

Implements biological memory mechanisms:
- Pattern Separation (Dentate Gyrus)
- Pattern Completion (CA3)
- Memory Consolidation
"""

from nokai.hippocampus.memory import HippocampalMemory
from nokai.hippocampus.pattern_separation import PatternSeparator
from nokai.hippocampus.retrieval import MemoryRetrieval

__all__ = ["HippocampalMemory", "PatternSeparator", "MemoryRetrieval"]
