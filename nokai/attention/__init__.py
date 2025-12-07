"""
Attention Controller - Dynamic Resource Allocation

Biological Parallel:
    The brain dynamically allocates computational resources based on:
    
    1. Salience: Novel or important stimuli get more processing
    2. Task demands: Complex tasks activate more neural circuits
    3. Arousal: Alertness level affects processing depth
    4. Motivation: Dopamine modulates attentional focus
    
Implementation:
    The attention controller:
    1. Monitors current processing demands
    2. Allocates compute resources dynamically
    3. Controls which modules are active (sparsity)
    4. Implements energy-efficient processing
    
Efficiency:
    - O(1) resource allocation decisions
    - Enables significant compute savings
    - Adapts to task complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import time


class ProcessingMode(Enum):
    """Processing intensity modes."""
    MINIMAL = "minimal"         # Only essential modules
    STANDARD = "standard"       # Normal processing
    INTENSIVE = "intensive"     # All modules, full capacity
    EMERGENCY = "emergency"     # Maximum resources


@dataclass
class ResourceAllocation:
    """
    Resource allocation for current processing step.
    
    Each field represents a fraction of maximum capacity [0, 1].
    """
    cortex: float = 1.0
    hippocampus: float = 0.5
    prefrontal: float = 0.5
    thalamus: float = 1.0
    limbic: float = 0.3
    oscillations: float = 0.5
    semantic_memory: float = 0.2
    
    # Compute parameters
    attention_heads_fraction: float = 1.0
    sequence_length_fraction: float = 1.0
    batch_processing: bool = True
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'cortex': self.cortex,
            'hippocampus': self.hippocampus,
            'prefrontal': self.prefrontal,
            'thalamus': self.thalamus,
            'limbic': self.limbic,
            'oscillations': self.oscillations,
            'semantic_memory': self.semantic_memory,
            'attention_heads_fraction': self.attention_heads_fraction,
            'sequence_length_fraction': self.sequence_length_fraction,
        }


class AttentionController(nn.Module):
    """
    Dynamic Resource Allocation Controller
    
    Biological Parallel:
        The brain's attentional system (frontoparietal network) 
        decides where to focus processing resources.
        
        Key mechanisms:
        1. Bottom-up: Salience drives attention (novel stimuli)
        2. Top-down: Goals drive attention (task-relevant)
        3. Arousal: Overall activation level
        4. Fatigue: Processing efficiency decreases over time
        
    Implementation:
        We implement this as a controller that:
        1. Assesses input complexity and novelty
        2. Considers task demands and dopamine level
        3. Allocates resources to each module
        4. Tracks energy budget for efficiency
        
    Efficiency:
        - Major efficiency gains through sparsity
        - Only activated modules consume compute
        - Dynamic scaling based on need
    """
    
    def __init__(
        self,
        state_dim: int,
        num_modules: int = 7,
        hidden_dim: int = 128,
        energy_budget: float = 1.0,  # Normalized budget
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_modules = num_modules
        self.energy_budget = energy_budget
        
        # ============================================
        # SALIENCE DETECTOR (Bottom-up attention)
        # ============================================
        # Detects novel or important inputs
        self.salience_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # COMPLEXITY ESTIMATOR
        # ============================================
        # Estimates how much processing is needed
        self.complexity_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # RESOURCE ALLOCATOR
        # ============================================
        # Decides allocation for each module
        self.allocator = nn.Sequential(
            nn.Linear(state_dim + 3, hidden_dim),  # +3 for salience, complexity, dopamine
            nn.GELU(),
            nn.Linear(hidden_dim, num_modules),
            nn.Sigmoid(),
        )
        
        # ============================================
        # MODULE-SPECIFIC GATES
        # ============================================
        # Fine-grained control for each module
        self.module_gates = nn.ModuleList([
            nn.Linear(state_dim, 1) for _ in range(num_modules)
        ])
        
        # ============================================
        # ENERGY TRACKING
        # ============================================
        self.register_buffer('energy_spent', torch.tensor(0.0))
        self.register_buffer('energy_saved', torch.tensor(0.0))
        self.register_buffer('total_steps', torch.tensor(0, dtype=torch.long))
        
        # ============================================
        # HISTORY FOR ADAPTATION
        # ============================================
        self.register_buffer('salience_history', torch.zeros(100))
        self.register_buffer('complexity_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        
        # Module names for interpretability
        self.module_names = [
            'cortex', 'hippocampus', 'prefrontal', 
            'thalamus', 'limbic', 'oscillations', 'semantic'
        ]
        
        # Minimum allocation thresholds (some modules always run)
        self.min_allocations = torch.tensor([
            0.3,  # cortex - always need some
            0.1,  # hippocampus
            0.1,  # prefrontal
            0.5,  # thalamus - always filter
            0.1,  # limbic
            0.2,  # oscillations
            0.05, # semantic
        ])
    
    def forward(
        self,
        state: torch.Tensor,
        dopamine_level: float = 0.5,
        task_complexity: Optional[float] = None,
        force_mode: Optional[ProcessingMode] = None,
    ) -> Tuple[ResourceAllocation, Dict]:
        """
        Compute resource allocation for current state.
        
        Args:
            state: Current internal state [batch, dim] or [dim]
            dopamine_level: Current dopamine level
            task_complexity: Optional override for complexity
            force_mode: Optional forced processing mode
            
        Returns:
            allocation: Resource allocation for each module
            metadata: Allocation details
        """
        if state.dim() == 3:
            # Average over sequence
            state = state.mean(dim=1)
        elif state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        
        # Handle forced mode
        if force_mode is not None:
            allocation = self._get_forced_allocation(force_mode)
            return allocation, {'mode': force_mode.value, 'forced': True}
        
        # ============================================
        # COMPUTE SALIENCE
        # ============================================
        salience = self.salience_net(state)  # [batch, 1]
        
        # ============================================
        # COMPUTE COMPLEXITY
        # ============================================
        if task_complexity is not None:
            complexity = torch.tensor([[task_complexity]], device=state.device)
            complexity = complexity.expand(batch_size, 1)
        else:
            complexity = self.complexity_net(state)
        
        # ============================================
        # COMBINE INPUTS
        # ============================================
        dopamine_t = torch.tensor([[dopamine_level]], device=state.device)
        dopamine_t = dopamine_t.expand(batch_size, 1)
        
        combined = torch.cat([state, salience, complexity, dopamine_t], dim=-1)
        
        # ============================================
        # COMPUTE BASE ALLOCATIONS
        # ============================================
        allocations = self.allocator(combined)  # [batch, num_modules]
        
        # ============================================
        # APPLY MODULE-SPECIFIC GATES
        # ============================================
        gates = []
        for gate in self.module_gates:
            g = torch.sigmoid(gate(state))
            gates.append(g)
        gates = torch.cat(gates, dim=-1)  # [batch, num_modules]
        
        # Combine allocations with gates
        final_allocations = allocations * gates
        
        # Apply minimum thresholds
        min_alloc = self.min_allocations.to(state.device).unsqueeze(0)
        final_allocations = torch.maximum(final_allocations, min_alloc)
        
        # ============================================
        # ENERGY BUDGET ENFORCEMENT
        # ============================================
        total_allocation = final_allocations.sum(dim=-1, keepdim=True)
        if (total_allocation > self.energy_budget * self.num_modules).any():
            # Scale down to budget
            scale = (self.energy_budget * self.num_modules) / total_allocation
            final_allocations = final_allocations * scale
            # Re-apply minimums
            final_allocations = torch.maximum(final_allocations, min_alloc)
        
        # ============================================
        # UPDATE STATISTICS
        # ============================================
        self.total_steps += 1
        self.energy_spent += final_allocations.sum().item() / batch_size
        self.energy_saved += (self.num_modules - final_allocations.sum().item() / batch_size)
        
        # Update history
        ptr = self.history_ptr.item()
        self.salience_history[ptr] = salience.mean().item()
        self.complexity_history[ptr] = complexity.mean().item()
        self.history_ptr = (self.history_ptr + 1) % 100
        
        # ============================================
        # CREATE ALLOCATION OBJECT
        # ============================================
        mean_allocs = final_allocations.mean(dim=0)
        allocation = ResourceAllocation(
            cortex=mean_allocs[0].item(),
            hippocampus=mean_allocs[1].item(),
            prefrontal=mean_allocs[2].item(),
            thalamus=mean_allocs[3].item(),
            limbic=mean_allocs[4].item(),
            oscillations=mean_allocs[5].item(),
            semantic_memory=mean_allocs[6].item() if self.num_modules > 6 else 0.2,
            attention_heads_fraction=min(1.0, 0.5 + complexity.mean().item()),
            sequence_length_fraction=min(1.0, 0.5 + 0.5 * salience.mean().item()),
        )
        
        metadata = {
            'salience': salience.mean().item(),
            'complexity': complexity.mean().item(),
            'total_allocation': final_allocations.sum(dim=-1).mean().item(),
            'per_module': {
                name: mean_allocs[i].item() 
                for i, name in enumerate(self.module_names[:self.num_modules])
            },
            'energy_efficiency': self.get_efficiency(),
            'mode': self._determine_mode(allocation),
        }
        
        return allocation, metadata
    
    def _get_forced_allocation(self, mode: ProcessingMode) -> ResourceAllocation:
        """Get allocation for forced mode."""
        if mode == ProcessingMode.MINIMAL:
            return ResourceAllocation(
                cortex=0.3, hippocampus=0.0, prefrontal=0.1,
                thalamus=0.5, limbic=0.1, oscillations=0.2,
                semantic_memory=0.0, attention_heads_fraction=0.25,
            )
        elif mode == ProcessingMode.STANDARD:
            return ResourceAllocation()  # defaults
        elif mode == ProcessingMode.INTENSIVE:
            return ResourceAllocation(
                cortex=1.0, hippocampus=1.0, prefrontal=1.0,
                thalamus=1.0, limbic=1.0, oscillations=1.0,
                semantic_memory=1.0, attention_heads_fraction=1.0,
            )
        else:  # EMERGENCY
            return ResourceAllocation(
                cortex=1.0, hippocampus=1.0, prefrontal=1.0,
                thalamus=1.0, limbic=1.0, oscillations=1.0,
                semantic_memory=1.0, attention_heads_fraction=1.0,
                batch_processing=False,  # Process individually for accuracy
            )
    
    def _determine_mode(self, allocation: ResourceAllocation) -> str:
        """Determine effective processing mode from allocation."""
        total = (allocation.cortex + allocation.hippocampus + 
                allocation.prefrontal + allocation.limbic)
        
        if total < 1.0:
            return ProcessingMode.MINIMAL.value
        elif total < 2.5:
            return ProcessingMode.STANDARD.value
        else:
            return ProcessingMode.INTENSIVE.value
    
    def get_efficiency(self) -> float:
        """Get current energy efficiency ratio."""
        if self.total_steps == 0:
            return 1.0
        total_possible = self.total_steps.item() * self.num_modules
        return self.energy_saved.item() / total_possible if total_possible > 0 else 1.0
    
    def should_activate_module(
        self,
        module_name: str,
        allocation: ResourceAllocation,
        threshold: float = 0.2,
    ) -> bool:
        """
        Determine if a specific module should be activated.
        
        Args:
            module_name: Name of the module
            allocation: Current resource allocation
            threshold: Minimum allocation to activate
            
        Returns:
            True if module should be activated
        """
        alloc_dict = allocation.to_dict()
        module_alloc = alloc_dict.get(module_name, 0.0)
        return module_alloc >= threshold
    
    def get_active_modules(
        self,
        allocation: ResourceAllocation,
        threshold: float = 0.2,
    ) -> List[str]:
        """Get list of modules that should be active."""
        active = []
        alloc_dict = allocation.to_dict()
        
        for name in self.module_names:
            if alloc_dict.get(name, 0.0) >= threshold:
                active.append(name)
        
        return active
    
    def energy_check(self, state: torch.Tensor) -> bool:
        """Quick check if full attention control is needed."""
        # Always run for novel/complex inputs
        variance = state.var(dim=-1).mean().item() if state.dim() > 1 else state.var().item()
        return variance > 0.3


class AdaptiveCompute(nn.Module):
    """
    Adaptive Computation Time (ACT) - Think Longer for Hard Problems
    
    Biological Parallel:
        The brain thinks longer about harder problems. This module
        implements variable computation depth based on difficulty.
    """
    
    def __init__(
        self,
        dim: int,
        max_iterations: int = 5,
        threshold: float = 0.99,
    ):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.threshold = threshold
        
        # Halting probability
        self.halt_prob = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Processing block (can be replaced with any computation)
        self.processor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process with adaptive computation time.
        
        Harder inputs get more processing iterations.
        """
        batch_size = x.shape[0] if x.dim() > 1 else 1
        
        # Initialize
        state = x
        cumulative_halt = torch.zeros(batch_size, 1, device=x.device)
        remainder = torch.ones(batch_size, 1, device=x.device)
        outputs = torch.zeros_like(x)
        
        iterations = 0
        
        for i in range(self.max_iterations):
            # Process
            processed = self.processor(state)
            
            # Compute halting probability
            halt_prob = self.halt_prob(processed)
            
            # Update cumulative halt
            cumulative_halt = cumulative_halt + halt_prob * remainder
            remainder = remainder * (1 - halt_prob)
            
            # Accumulate output
            outputs = outputs + halt_prob * processed
            
            iterations = i + 1
            
            # Check if all samples have halted
            if (cumulative_halt > self.threshold).all():
                break
            
            state = processed
        
        # Handle remainder
        outputs = outputs + remainder * state
        
        metadata = {
            'iterations': iterations,
            'mean_halt_prob': cumulative_halt.mean().item(),
        }
        
        return outputs, metadata
