"""
Striatum - Action Selection and Decision Making

Biological Parallel:
    The striatum (part of basal ganglia) is the brain's action selector.
    It receives input from:
    - Cortex (possible actions)
    - VTA (dopamine/reward signals)
    - Hippocampus (memory context)
    
    Two pathways:
    - Direct pathway: "Go" - promotes selected action
    - Indirect pathway: "NoGo" - inhibits competing actions
    
    The balance between these pathways determines action selection.

Implementation:
    We model this as a cost/benefit analyzer that:
    1. Evaluates multiple action candidates
    2. Computes expected value (benefit) of each
    3. Computes expected cost (risk/effort) of each
    4. Selects based on net value, modulated by dopamine
    
Efficiency:
    - O(A) complexity where A = number of action candidates
    - Sparse activation for efficiency
    - Memory-efficient action history
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math


@dataclass
class ActionCandidate:
    """
    Represents a possible action with its evaluation.
    
    Biological Mapping:
        - action_id: Cortical representation of the action
        - expected_benefit: Predicted reward (nucleus accumbens)
        - expected_cost: Predicted effort/risk (anterior insula)
        - confidence: Certainty of prediction (prefrontal cortex)
    """
    action_id: int
    action_embedding: torch.Tensor
    expected_benefit: float
    expected_cost: float
    confidence: float
    net_value: float = 0.0


class StriatumSelector(nn.Module):
    """
    The Striatum - Decision Making Under Uncertainty
    
    Biological Parallel:
        The striatum integrates cortical "proposals" with limbic 
        "motivation" to select actions. Key mechanisms:
        
        1. Direct Pathway (D1 receptors):
           - Activated by dopamine
           - Promotes movement/action
           - "GO" signal
           
        2. Indirect Pathway (D2 receptors):
           - Inhibited by dopamine
           - Suppresses competing actions
           - "NO-GO" signal
           
        The balance is modulated by dopamine level:
        - High DA → Bias toward GO (more action, risk-taking)
        - Low DA → Bias toward NO-GO (less action, risk-aversion)
    
    Implementation:
        We compute action values with explicit cost/benefit analysis,
        then apply dopamine-modulated selection.
    
    Efficiency:
        - Sparse candidate evaluation
        - O(A) for A action candidates
        - Energy-efficient gating
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_action_candidates: int = 16,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_candidates = num_action_candidates
        self.temperature = temperature
        
        # ============================================
        # DIRECT PATHWAY (D1) - "GO" Circuit
        # ============================================
        # Evaluates expected benefits of actions
        self.direct_pathway = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Benefit score
        )
        
        # ============================================
        # INDIRECT PATHWAY (D2) - "NO-GO" Circuit
        # ============================================
        # Evaluates expected costs/risks of actions
        self.indirect_pathway = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Cost score
        )
        
        # ============================================
        # CONFIDENCE ESTIMATION
        # ============================================
        # How certain are we about this decision?
        self.confidence_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Confidence [0, 1]
        )
        
        # ============================================
        # ACTION GENERATOR
        # ============================================
        # Generates candidate actions from state
        self.action_generator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_action_candidates * action_dim),
        )
        
        # Buffer for tracking action history (for learning)
        self.register_buffer('action_history', torch.zeros(1000, action_dim))
        self.register_buffer('value_history', torch.zeros(1000))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        
        # Dopamine modulation weights
        self.da_direct_weight = nn.Parameter(torch.tensor(1.0))
        self.da_indirect_weight = nn.Parameter(torch.tensor(-0.5))
        
        # Exploration noise (annealed during training)
        self.exploration_noise = 0.1
        
        # Synaptic weight tracking for plasticity
        self.register_buffer('synaptic_weights', torch.ones(num_action_candidates))
    
    def generate_candidates(
        self, 
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate action candidates from current state.
        
        Args:
            state: Current internal state [batch, state_dim]
            
        Returns:
            candidates: Action embeddings [batch, num_candidates, action_dim]
        """
        batch_size = state.shape[0]
        
        # Generate raw candidates
        raw = self.action_generator(state)  # [batch, num_candidates * action_dim]
        candidates = raw.view(batch_size, self.num_candidates, self.action_dim)
        
        # Add exploration noise
        if self.training and self.exploration_noise > 0:
            noise = torch.randn_like(candidates) * self.exploration_noise
            candidates = candidates + noise
        
        return candidates
    
    def evaluate_candidate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dopamine_level: float = 0.5,
    ) -> Tuple[float, float, float]:
        """
        Evaluate a single action candidate.
        
        Args:
            state: Current state [batch, state_dim]
            action: Action embedding [batch, action_dim]
            dopamine_level: Current dopamine level [0, 1]
            
        Returns:
            benefit: Expected benefit
            cost: Expected cost
            net_value: Dopamine-modulated net value
        """
        # Combine state and action
        combined = torch.cat([state, action], dim=-1)
        
        # Direct pathway: Expected benefit
        benefit = self.direct_pathway(combined).squeeze(-1)
        
        # Indirect pathway: Expected cost
        cost = self.indirect_pathway(combined).squeeze(-1)
        
        # Dopamine modulation
        # High DA → amplify benefit (D1), suppress cost sensitivity (D2)
        da_effect = dopamine_level - 0.5  # Center at 0
        modulated_benefit = benefit * (1 + self.da_direct_weight * da_effect)
        modulated_cost = cost * (1 + self.da_indirect_weight * da_effect)
        
        # Net value
        net_value = modulated_benefit - modulated_cost
        
        return benefit.mean().item(), cost.mean().item(), net_value.mean().item()
    
    def forward(
        self,
        state: torch.Tensor,
        dopamine_level: float = 0.5,
        action_candidates: Optional[torch.Tensor] = None,
        return_all_values: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Select an action based on current state and dopamine level.
        
        Biological Process:
            1. Cortex proposes action candidates
            2. Direct pathway evaluates benefits
            3. Indirect pathway evaluates costs
            4. Dopamine modulates the GO/NO-GO balance
            5. Winner-take-all selection
        
        Args:
            state: Current internal state [batch, state_dim]
            dopamine_level: Current dopamine level [0, 1]
            action_candidates: Optional pre-computed candidates
            return_all_values: Return values for all candidates
            
        Returns:
            selected_action: Chosen action embedding [batch, action_dim]
            metadata: Selection details
        """
        batch_size = state.shape[0]
        
        # Generate candidates if not provided
        if action_candidates is None:
            candidates = self.generate_candidates(state)
        else:
            candidates = action_candidates
        
        # Evaluate all candidates
        benefits = []
        costs = []
        net_values = []
        
        for i in range(self.num_candidates):
            action = candidates[:, i, :]
            combined = torch.cat([state, action], dim=-1)
            
            # Direct pathway
            benefit = self.direct_pathway(combined)
            benefits.append(benefit)
            
            # Indirect pathway
            cost = self.indirect_pathway(combined)
            costs.append(cost)
        
        benefits = torch.cat(benefits, dim=-1)  # [batch, num_candidates]
        costs = torch.cat(costs, dim=-1)
        
        # Dopamine modulation
        da_effect = dopamine_level - 0.5
        modulated_benefits = benefits * (1 + self.da_direct_weight * da_effect)
        modulated_costs = costs * (1 + self.da_indirect_weight * da_effect)
        
        # Net values with synaptic weight modulation (plasticity)
        net_values = modulated_benefits - modulated_costs
        net_values = net_values * self.synaptic_weights.unsqueeze(0)
        
        # Confidence estimation
        confidence = self.confidence_net(state)
        
        # Selection via softmax (soft winner-take-all)
        selection_probs = F.softmax(net_values / self.temperature, dim=-1)
        
        # Select action (differentiable selection using weighted sum)
        selected_action = torch.einsum('bc,bcd->bd', selection_probs, candidates)
        
        # Also get hard selection for analysis
        hard_selection = net_values.argmax(dim=-1)
        
        # Update synaptic weights based on selection frequency
        # (Hebbian: frequently selected actions strengthen)
        if self.training:
            for idx in hard_selection.tolist():
                self.synaptic_weights[idx] = min(
                    self.synaptic_weights[idx] * 1.001, 
                    2.0
                )
        
        # Store in history
        ptr = self.history_ptr.item()
        self.action_history[ptr] = selected_action.detach().mean(0)
        self.value_history[ptr] = net_values.max(dim=-1).values.mean().detach()
        self.history_ptr = (self.history_ptr + 1) % 1000
        
        metadata = {
            'selected_action_idx': hard_selection.tolist(),
            'max_net_value': net_values.max(dim=-1).values.mean().item(),
            'mean_benefit': benefits.mean().item(),
            'mean_cost': costs.mean().item(),
            'confidence': confidence.mean().item(),
            'selection_entropy': -(selection_probs * selection_probs.log()).sum(-1).mean().item(),
            'dopamine_effect': da_effect,
            'synaptic_weights_mean': self.synaptic_weights.mean().item(),
        }
        
        if return_all_values:
            metadata['all_benefits'] = benefits.detach()
            metadata['all_costs'] = costs.detach()
            metadata['all_net_values'] = net_values.detach()
            metadata['selection_probs'] = selection_probs.detach()
        
        return selected_action, metadata
    
    def energy_check(self, state: torch.Tensor) -> bool:
        """
        Check if this module should activate (Sparsity/Efficiency).
        
        Biological Parallel:
            Not every stimulus requires a decision. The striatum
            should only activate when action is needed.
            
        Returns:
            True if decision-making is warranted, False otherwise
        """
        # Simple energy check: Is there significant state change?
        state_magnitude = state.abs().mean().item()
        
        # Threshold for activation
        return state_magnitude > 0.1
    
    def apply_plasticity(
        self,
        selected_idx: int,
        reward: float,
        learning_rate: float = 0.01,
    ):
        """
        Apply plasticity to synaptic weights based on outcome.
        
        Biological Parallel:
            Actions followed by reward have their "pathways" strengthened.
            This is dopamine-dependent LTP in the striatum.
        """
        with torch.no_grad():
            if reward > 0:
                # LTP: Strengthen selected pathway
                self.synaptic_weights[selected_idx] = min(
                    self.synaptic_weights[selected_idx] + learning_rate * reward,
                    2.0
                )
            else:
                # LTD: Weaken if negative outcome
                self.synaptic_weights[selected_idx] = max(
                    self.synaptic_weights[selected_idx] + learning_rate * reward,
                    0.5
                )
    
    def reset_exploration(self, new_noise: float = 0.1):
        """Reset exploration parameters."""
        self.exploration_noise = new_noise
    
    def anneal_exploration(self, factor: float = 0.99):
        """Decrease exploration over time."""
        self.exploration_noise *= factor


class ProceduralSkillMemory(nn.Module):
    """
    Basal Ganglia Skill Storage - "Muscle Memory"
    
    Biological Parallel:
        The basal ganglia stores procedural memories - automatic
        skills that don't require conscious attention:
        - Motor sequences
        - Habitual responses
        - Learned procedures
        
    Implementation:
        A memory bank of state→action mappings that can be:
        1. Learned slowly through repetition
        2. Retrieved quickly without deliberation
        3. Chunked into sequences
    
    Efficiency:
        - O(1) retrieval once learned
        - Sparse storage (only frequently used skills)
        - Memory-mapped for large skill libraries
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_skills: int = 1000,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_skills = num_skills
        self.chunk_size = chunk_size
        
        # Skill key encoder
        self.key_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        
        # Skill storage
        self.register_buffer('skill_keys', torch.zeros(num_skills, state_dim))
        self.register_buffer('skill_values', torch.zeros(num_skills, chunk_size, action_dim))
        self.register_buffer('skill_strengths', torch.zeros(num_skills))  # Synaptic weights
        self.register_buffer('skill_count', torch.tensor(0, dtype=torch.long))
        
        # Retrieval temperature
        self.retrieval_temperature = 0.1
    
    def store_skill(
        self,
        state: torch.Tensor,
        action_sequence: torch.Tensor,
    ):
        """
        Store a new skill (state→action mapping).
        
        Args:
            state: Trigger state [state_dim]
            action_sequence: Sequence of actions [chunk_size, action_dim]
        """
        if self.skill_count >= self.num_skills:
            # Replace weakest skill
            idx = self.skill_strengths.argmin().item()
        else:
            idx = self.skill_count.item()
            self.skill_count += 1
        
        key = self.key_encoder(state.unsqueeze(0)).squeeze(0)
        self.skill_keys[idx] = key.detach()
        
        # Pad or truncate action sequence
        seq_len = min(action_sequence.shape[0], self.chunk_size)
        self.skill_values[idx, :seq_len] = action_sequence[:seq_len].detach()
        
        # Initialize strength
        self.skill_strengths[idx] = 1.0
    
    def retrieve_skill(
        self,
        state: torch.Tensor,
        threshold: float = 0.7,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a matching skill if one exists.
        
        Args:
            state: Current state [batch, state_dim]
            threshold: Minimum similarity for retrieval
            
        Returns:
            Action sequence if skill found, None otherwise
        """
        if self.skill_count == 0:
            return None
        
        # Encode query
        query = self.key_encoder(state)  # [batch, state_dim]
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query.unsqueeze(1),
            self.skill_keys[:self.skill_count].unsqueeze(0),
            dim=-1
        )  # [batch, num_stored_skills]
        
        # Find best match
        best_similarity, best_idx = similarities.max(dim=-1)
        
        if best_similarity.mean().item() > threshold:
            # Retrieve and strengthen (LTP)
            self.skill_strengths[best_idx] = min(
                self.skill_strengths[best_idx] * 1.01,
                10.0
            )
            return self.skill_values[best_idx]
        
        return None
    
    def energy_check(self) -> bool:
        """Check if skill retrieval should be attempted."""
        return self.skill_count > 0
