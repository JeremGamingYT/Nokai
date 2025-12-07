"""
Hebbian Learning - "Neurons that fire together, wire together"

Implements biologically-plausible learning rules including:
- Basic Hebbian rule
- Oja's rule (with normalization)
- STDP (Spike-Timing Dependent Plasticity)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class HebbianPlasticity(nn.Module):
    """
    Hebbian learning with Oja's normalization.
    
    Update rule: Δw = η * (post * pre - α * post² * w)
    
    The second term prevents unbounded weight growth.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.0001,
        normalize: bool = True,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.normalize = normalize
    
    def compute_update(
        self,
        weight: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weight update based on pre/post activations.
        
        Args:
            weight: Current weight matrix [out, in]
            pre: Pre-synaptic activations [batch, in]
            post: Post-synaptic activations [batch, out]
            
        Returns:
            Weight update delta
        """
        # Average over batch
        pre_mean = pre.mean(0)
        post_mean = post.mean(0)
        
        # Hebbian term: outer product of post and pre
        hebbian = torch.outer(post_mean, pre_mean)
        
        if self.normalize:
            # Oja's rule: prevent unbounded growth
            oja_term = (post_mean ** 2).unsqueeze(1) * weight
            delta = self.learning_rate * (hebbian - oja_term)
        else:
            delta = self.learning_rate * hebbian
        
        # Weight decay
        delta = delta - self.decay_rate * weight
        
        return delta
    
    def apply_update(
        self,
        weight: nn.Parameter,
        pre: torch.Tensor,
        post: torch.Tensor,
    ):
        """Apply Hebbian update to weight in-place."""
        with torch.no_grad():
            delta = self.compute_update(weight.data, pre, post)
            weight.data.add_(delta)


class STDPRule(nn.Module):
    """
    Spike-Timing Dependent Plasticity
    
    Implements asymmetric learning window:
    - Pre before Post → LTP (Long-Term Potentiation)
    - Post before Pre → LTD (Long-Term Depression)
    
    Simplified version uses activation magnitudes as proxy for spike timing.
    """
    
    def __init__(
        self,
        ltp_rate: float = 0.001,  # Potentiation rate
        ltd_rate: float = 0.0005,  # Depression rate
        time_constant: float = 20.0,  # ms
    ):
        super().__init__()
        self.ltp_rate = ltp_rate
        self.ltd_rate = ltd_rate
        self.time_constant = time_constant
        
        # Traces for temporal integration
        self.pre_trace = None
        self.post_trace = None
    
    def update_traces(self, pre: torch.Tensor, post: torch.Tensor, dt: float = 1.0):
        """Update eligibility traces."""
        decay = torch.exp(torch.tensor(-dt / self.time_constant))
        
        if self.pre_trace is None:
            self.pre_trace = pre.clone()
            self.post_trace = post.clone()
        else:
            self.pre_trace = decay * self.pre_trace + (1 - decay) * pre
            self.post_trace = decay * self.post_trace + (1 - decay) * post
    
    def compute_update(
        self,
        weight: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Compute STDP weight update."""
        self.update_traces(pre, post)
        
        # LTP: pre trace correlates with current post
        ltp = torch.outer(post.mean(0), self.pre_trace.mean(0))
        
        # LTD: post trace correlates with current pre
        ltd = torch.outer(self.post_trace.mean(0), pre.mean(0))
        
        delta = self.ltp_rate * ltp - self.ltd_rate * ltd
        
        return delta
    
    def reset_traces(self):
        """Reset eligibility traces."""
        self.pre_trace = None
        self.post_trace = None


class LocalLearningLayer(nn.Module):
    """
    A layer that can learn via local Hebbian rules without backprop.
    
    Combines feedforward computation with local learning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rule: str = "hebbian",
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # Learning rule
        if learning_rule == "stdp":
            self.plasticity = STDPRule(ltp_rate=learning_rate)
        else:
            self.plasticity = HebbianPlasticity(learning_rate=learning_rate)
        
        # Store activations for learning
        self.last_input = None
        self.last_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        
        # Store for learning
        if self.training:
            self.last_input = x.detach()
            self.last_output = output.detach()
        
        return output
    
    def local_update(self):
        """Apply local learning update."""
        if self.last_input is not None and self.last_output is not None:
            with torch.no_grad():
                delta = self.plasticity.compute_update(
                    self.linear.weight.data,
                    self.last_input,
                    self.last_output,
                )
                self.linear.weight.data.add_(delta)
