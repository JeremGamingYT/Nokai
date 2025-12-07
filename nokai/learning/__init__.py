"""
Learning Module - Bio-inspired learning rules

Implements Hebbian plasticity and predictive coding as
alternatives to pure backpropagation.
"""

from nokai.learning.hebbian import HebbianPlasticity, STDPRule
from nokai.learning.predictive import PredictiveCodingLayer

__all__ = ["HebbianPlasticity", "STDPRule", "PredictiveCodingLayer"]
