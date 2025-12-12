"""
Structural Plasticity - Dynamic Topology

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

The brain's wiring is not fixed! Synapses are created and destroyed constantly:

1. SYNAPTOGENESIS: New synapses form between frequently co-active neurons
2. PRUNING: Unused/weak synapses are eliminated
3. NEUROGENESIS: New neurons are born (especially in hippocampus)
4. APOPTOSIS: Inactive neurons are removed

This allows the brain to:
- Adapt to new tasks
- Become more efficient
- Form specialized modules
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math


class StructuralPlasticity(nn.Module):
    """
    Structural plasticity: dynamic creation and pruning of connections.
    """
    
    def __init__(
        self,
        create_threshold: float = 0.8,  # Correlation threshold for synaptogenesis
        prune_threshold: float = 0.01,  # Weight threshold for pruning
        max_synapses_per_neuron: int = 100,
        min_synapses_per_neuron: int = 10,
    ):
        super().__init__()
        
        self.create_threshold = create_threshold
        self.prune_threshold = prune_threshold
        self.max_synapses = max_synapses_per_neuron
        self.min_synapses = min_synapses_per_neuron
        
        # Statistics
        self.register_buffer('total_created', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_pruned', torch.tensor(0, dtype=torch.long))
    
    def compute_correlations(
        self,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute correlation matrix from activations.
        
        Args:
            activations: [batch, neurons] activation tensor
            
        Returns:
            correlations: [neurons, neurons] correlation matrix
        """
        # Normalize
        centered = activations - activations.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, keepdim=True).clamp(min=1e-8)
        normalized = centered / std
        
        # Correlation
        corr = torch.matmul(normalized.T, normalized) / activations.shape[0]
        
        return corr
    
    def apply_synaptogenesis(
        self,
        weight: torch.Tensor,
        correlations: torch.Tensor,
        init_strength: float = 0.01,
    ) -> Tuple[torch.Tensor, int]:
        """
        Create new synapses between correlated neurons.
        
        Args:
            weight: Current weight matrix [out, in]
            correlations: Correlation matrix [neurons, neurons]
            init_strength: Initial weight for new synapses
            
        Returns:
            updated_weight: Weight matrix with new synapses
            num_created: Number of synapses created
        """
        # Find highly correlated pairs without existing connections
        high_corr = correlations.abs() > self.create_threshold
        no_synapse = weight.abs() < 1e-10
        
        # Ensure we don't exceed max synapses per neuron
        current_count = (weight.abs() > 1e-10).sum(dim=1)
        has_room = current_count < self.max_synapses
        
        # Candidates for creation
        candidates = high_corr & no_synapse
        
        # Create new synapses
        num_created = 0
        with torch.no_grad():
            for i in range(weight.shape[0]):
                if not has_room[i]:
                    continue
                
                # Find best candidates for this row
                row_candidates = candidates[i] if i < candidates.shape[0] else torch.zeros(weight.shape[1], dtype=torch.bool, device=weight.device)
                n_to_create = min(
                    row_candidates.sum().item(),
                    self.max_synapses - current_count[i].item()
                )
                
                if n_to_create > 0:
                    # Get top candidates by correlation
                    if i < correlations.shape[0]:
                        corr_row = correlations[i] * row_candidates.float()
                    else:
                        corr_row = torch.zeros(weight.shape[1], device=weight.device)
                    
                    _, top_idx = torch.topk(corr_row.abs(), min(n_to_create, len(corr_row)))
                    
                    # Create with sign from correlation
                    for idx in top_idx:
                        if idx < weight.shape[1] and i < correlations.shape[0]:
                            weight[i, idx] = init_strength * torch.sign(correlations[i, idx])
                            num_created += 1
        
        self.total_created += num_created
        return weight, num_created
    
    def apply_pruning(
        self,
        weight: torch.Tensor,
        activity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Remove weak/unused synapses.
        
        Args:
            weight: Weight matrix [out, in]
            activity: Optional activity metric [out] or [out, in]
            
        Returns:
            updated_weight: Pruned weight matrix
            num_pruned: Number of synapses removed
        """
        # Find weak weights
        weak_weights = weight.abs() < self.prune_threshold
        
        # Ensure minimum synapses per neuron
        current_count = (weight.abs() >= self.prune_threshold).sum(dim=1)
        
        # Only prune where we have enough synapses
        can_prune = current_count > self.min_synapses
        
        # Create pruning mask
        prune_mask = weak_weights.clone()
        for i in range(weight.shape[0]):
            if not can_prune[i]:
                prune_mask[i] = False
        
        # If activity provided, also prune inactive synapses
        if activity is not None:
            if activity.dim() == 1:
                # Per-neuron activity
                low_activity = activity < 0.01
                for i in range(weight.shape[0]):
                    if i < len(low_activity) and low_activity[i] and can_prune[i]:
                        # Prune random subset
                        n_to_prune = max(0, current_count[i].item() - self.min_synapses)
                        if n_to_prune > 0:
                            existing = (weight[i].abs() >= self.prune_threshold).nonzero().squeeze(-1)
                            if len(existing) > self.min_synapses:
                                to_remove = existing[torch.randperm(len(existing))[:n_to_prune]]
                                prune_mask[i, to_remove] = True
        
        # Apply pruning
        num_pruned = prune_mask.sum().item()
        with torch.no_grad():
            weight[prune_mask] = 0
        
        self.total_pruned += num_pruned
        return weight, num_pruned
    
    def step(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> Dict:
        """
        Full structural plasticity step.
        
        Args:
            weight: Weight matrix to modify
            activations: Recent activations
            
        Returns:
            info: Statistics about changes
        """
        # Compute correlations
        correlations = self.compute_correlations(activations)
        
        # Synaptogenesis
        weight, created = self.apply_synaptogenesis(weight, correlations)
        
        # Pruning
        activity = activations.abs().mean(dim=0)
        weight, pruned = self.apply_pruning(weight, activity)
        
        # Sparsity
        sparsity = (weight.abs() < 1e-10).float().mean().item()
        
        return {
            'created': created,
            'pruned': pruned,
            'net_change': created - pruned,
            'sparsity': sparsity,
            'total_created': self.total_created.item(),
            'total_pruned': self.total_pruned.item(),
        }


class SynapticPruner(nn.Module):
    """
    Magnitude-based synaptic pruning with regrowth.
    
    Implements iterative magnitude pruning (IMP) with:
    - Gradual pruning schedule
    - Regrowth of pruned connections
    - Activity-based importance
    """
    
    def __init__(
        self,
        target_sparsity: float = 0.9,
        pruning_rate: float = 0.1,
        regrowth_rate: float = 0.05,
    ):
        super().__init__()
        
        self.target_sparsity = target_sparsity
        self.pruning_rate = pruning_rate
        self.regrowth_rate = regrowth_rate
    
    def prune_by_magnitude(
        self,
        weight: torch.Tensor,
        sparsity: float,
    ) -> torch.Tensor:
        """Prune smallest magnitude weights."""
        flat = weight.abs().flatten()
        k = int(flat.numel() * sparsity)
        
        if k >= flat.numel():
            return torch.zeros_like(weight)
        
        threshold = flat.kthvalue(k + 1).values
        
        mask = (weight.abs() >= threshold).float()
        return weight * mask
    
    def regrow_random(
        self,
        weight: torch.Tensor,
        mask: torch.Tensor,
        n_regrow: int,
    ) -> torch.Tensor:
        """Randomly regrow pruned connections."""
        pruned_idx = (mask == 0).nonzero()
        
        if len(pruned_idx) == 0:
            return weight
        
        # Random selection
        perm = torch.randperm(len(pruned_idx))[:n_regrow]
        regrow_idx = pruned_idx[perm]
        
        # Initialize new weights
        with torch.no_grad():
            for idx in regrow_idx:
                weight[tuple(idx)] = torch.randn(1, device=weight.device).item() * 0.01
        
        return weight
