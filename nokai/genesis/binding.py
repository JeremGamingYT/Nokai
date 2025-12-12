"""
Oscillatory Binding - Phase-Based Concept Linking

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

How does the brain bind "blue" and "apple" into "blue apple"?

Answer: OSCILLATORY SYNCHRONY

Concepts that belong together fire in the SAME PHASE of gamma oscillations,
nested within theta oscillations.

Example:
- "blue" neurons fire at phase 0° of gamma
- "apple" neurons fire at phase 0° of gamma  
- "car" neurons fire at phase 180° of gamma

Because blue and apple are in sync, they're perceived as bound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class PhaseCoding(nn.Module):
    """
    Phase-based encoding of concepts.
    
    Each concept has a preferred phase within the oscillatory cycle.
    Concepts with similar phases are "bound" together.
    """
    
    def __init__(
        self,
        num_units: int,
        initial_phases: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.num_units = num_units
        
        # Preferred phase for each unit
        if initial_phases is not None:
            self.phases = nn.Parameter(initial_phases)
        else:
            self.phases = nn.Parameter(torch.rand(num_units) * 2 * math.pi)
    
    def get_phase_vector(self, t: float) -> torch.Tensor:
        """
        Get activation modulation at time t.
        
        Units fire maximally when oscillation phase matches their preferred phase.
        
        Args:
            t: Current time (in radians or seconds)
            
        Returns:
            modulation: [num_units] in [0, 1]
        """
        # Current oscillation phase (assume some frequency)
        current_phase = t % (2 * math.pi)
        
        # Activation = cos(current - preferred), normalized to [0, 1]
        phase_diff = current_phase - self.phases
        modulation = 0.5 + 0.5 * torch.cos(phase_diff)
        
        return modulation
    
    def compute_binding_matrix(self) -> torch.Tensor:
        """
        Compute binding strength between all pairs of units.
        
        Units with similar phases are strongly bound.
        
        Returns:
            binding: [num_units, num_units] in [-1, 1]
        """
        phase_diff = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
        binding = torch.cos(phase_diff)
        return binding


class OscillatoryBinder(nn.Module):
    """
    Full oscillatory binding system.
    
    Implements:
    - Gamma oscillations (40Hz) for fast binding
    - Theta oscillations (6Hz) as envelope
    - Phase-based synchronization between concepts
    """
    
    def __init__(
        self,
        num_concepts: int,
        embedding_dim: int,
        theta_freq: float = 6.0,
        gamma_freq: float = 40.0,
        coupling_strength: float = 0.3,
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.coupling_strength = coupling_strength
        
        # Gamma phase per concept
        self.gamma_phases = nn.Parameter(
            torch.rand(num_concepts) * 2 * math.pi
        )
        
        # Phase coupling matrix (learned connectivity)
        self.coupling = nn.Parameter(
            torch.zeros(num_concepts, num_concepts)
        )
        
        # Concept embeddings
        self.embeddings = nn.Embedding(num_concepts, embedding_dim)
        
        # Current time
        self.register_buffer('t', torch.tensor(0.0))
    
    def step(self, dt: float = 0.001):
        """Advance time by dt seconds."""
        self.t = self.t + dt
        
        # Kuramoto-style phase dynamics
        with torch.no_grad():
            # Coupling effect
            phase_diff = self.gamma_phases.unsqueeze(0) - self.gamma_phases.unsqueeze(1)
            coupling_effect = self.coupling_strength * (
                self.coupling * torch.sin(phase_diff)
            ).sum(dim=1)
            
            # Phase update
            d_phase = 2 * math.pi * self.gamma_freq * dt + coupling_effect * dt
            self.gamma_phases.data = (self.gamma_phases.data + d_phase) % (2 * math.pi)
    
    def get_modulation(self, concept_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get current activation modulation for concepts.
        
        Args:
            concept_ids: Optional subset of concepts
            
        Returns:
            modulation: [num_concepts] or [len(concept_ids)]
        """
        # Theta envelope
        theta_phase = 2 * math.pi * self.theta_freq * self.t
        theta_mod = 0.5 + 0.5 * torch.cos(theta_phase)
        
        # Gamma fast oscillation
        if concept_ids is not None:
            gamma_phases = self.gamma_phases[concept_ids]
        else:
            gamma_phases = self.gamma_phases
        
        gamma_t = 2 * math.pi * self.gamma_freq * self.t
        gamma_mod = 0.5 + 0.5 * torch.cos(gamma_t - gamma_phases)
        
        # Gamma nested in theta
        return theta_mod * gamma_mod
    
    def bind(self, concept_a: int, concept_b: int, strength: float = 1.0):
        """
        Bind two concepts by synchronizing their phases.
        
        Args:
            concept_a, concept_b: Concept indices
            strength: Coupling strength (higher = tighter binding)
        """
        with torch.no_grad():
            # Strengthen coupling
            self.coupling.data[concept_a, concept_b] = strength
            self.coupling.data[concept_b, concept_a] = strength
            
            # Move phases closer
            mean_phase = (self.gamma_phases[concept_a] + self.gamma_phases[concept_b]) / 2
            self.gamma_phases.data[concept_a] = mean_phase
            self.gamma_phases.data[concept_b] = mean_phase
    
    def unbind(self, concept_a: int, concept_b: int):
        """Remove binding between concepts."""
        with torch.no_grad():
            self.coupling.data[concept_a, concept_b] = 0
            self.coupling.data[concept_b, concept_a] = 0
    
    def get_bound_concepts(self, concept_id: int, threshold: float = 0.7) -> List[int]:
        """
        Get concepts bound to a given concept.
        
        Args:
            concept_id: Target concept
            threshold: Phase similarity threshold
            
        Returns:
            List of bound concept IDs
        """
        target_phase = self.gamma_phases[concept_id]
        phase_diff = torch.cos(self.gamma_phases - target_phase)
        
        bound_mask = phase_diff > threshold
        bound_mask[concept_id] = False  # Exclude self
        
        return bound_mask.nonzero().squeeze(-1).tolist()
    
    def forward(
        self,
        concept_ids: torch.Tensor,
        return_binding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get modulated embeddings for concepts.
        
        Args:
            concept_ids: [batch] or [batch, seq] concept IDs
            return_binding: If True, return binding matrix
            
        Returns:
            embeddings: Phase-modulated embeddings
            binding: Optional binding matrix
        """
        # Get base embeddings
        emb = self.embeddings(concept_ids)
        
        # Get modulation
        flat_ids = concept_ids.view(-1)
        modulation = self.get_modulation(flat_ids)
        modulation = modulation.view(*concept_ids.shape, 1)
        
        # Apply modulation
        modulated_emb = emb * modulation
        
        # Advance time
        self.step()
        
        if return_binding:
            binding = self.compute_binding_matrix(concept_ids)
            return modulated_emb, binding
        
        return modulated_emb, None
    
    def compute_binding_matrix(self, concept_ids: torch.Tensor) -> torch.Tensor:
        """Compute binding matrix for given concepts."""
        phases = self.gamma_phases[concept_ids.view(-1)]
        n = phases.shape[0]
        
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
        binding = torch.cos(phase_diff)
        
        return binding


class AttentionWithBinding(nn.Module):
    """
    Attention mechanism enhanced with oscillatory binding.
    
    Standard attention notices individual tokens.
    This attention also biases toward bound tokens.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        binding_bias: float = 0.5,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.binding_bias = binding_bias
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        binding_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attention with optional binding bias.
        
        Args:
            x: Input [batch, seq, dim]
            binding_matrix: Optional [seq, seq] binding strengths
            
        Returns:
            output: [batch, seq, dim]
        """
        batch, seq, _ = x.shape
        
        # Project
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add binding bias
        if binding_matrix is not None:
            binding_bias = self.binding_bias * binding_matrix.unsqueeze(0).unsqueeze(0)
            attn = attn + binding_bias
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.dim)
        out = self.o_proj(out)
        
        return out
