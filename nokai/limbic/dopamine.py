"""
Dopamine Circuit - Reward Prediction and Motivation System

Biological Parallel:
    The Ventral Tegmental Area (VTA) releases dopamine based on 
    reward prediction errors (RPE). This signal modulates learning
    and attention across the brain.
    
    - Unexpected rewards → Dopamine burst → Enhanced learning
    - Expected rewards → No change → Baseline activity  
    - Missing expected rewards → Dopamine dip → Extinction

Implementation:
    We simulate dopamine as a scalar signal that modulates:
    1. Synaptic plasticity rates (higher DA = faster learning)
    2. Attention allocation (higher DA = more focus on current input)
    3. Memory encoding priority (higher DA = stronger encoding)

Efficiency:
    - O(1) per forward pass for dopamine computation
    - Sparse updates to avoid unnecessary recalculation
    - Memory-mapped history for large-scale analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class DopamineState:
    """
    Current state of the dopamine system.
    
    Biological Mapping:
        - level: Tonic dopamine (baseline mood/motivation)
        - burst: Phasic dopamine (reward-related spikes)
        - rpe: Reward Prediction Error (learning signal)
        - expected_reward: Internal prediction of upcoming reward
    """
    level: float = 0.5          # Baseline dopamine [0, 1]
    burst: float = 0.0          # Phasic response [-1, 1]
    rpe: float = 0.0            # Reward prediction error
    expected_reward: float = 0.0  # Value prediction


class RewardPredictionError(nn.Module):
    """
    Computes the Reward Prediction Error (RPE).
    
    Biological Parallel:
        The RPE is computed as: δ = R + γ*V(s') - V(s)
        Where:
            - R = actual reward received
            - V(s) = value estimate of current state
            - V(s') = value estimate of next state
            - γ = temporal discount factor
        
        This signal drives learning in the basal ganglia and cortex.
    
    Usage:
        The RPE modulates:
        - Hebbian learning rates (TD learning)
        - Attention allocation
        - Memory consolidation priority
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Value function approximator (Critic network)
        # Maps state representation to expected reward
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Reward prediction (learns expected rewards per context)
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Eligibility trace for TD learning
        self.register_buffer('eligibility_trace', torch.zeros(1))
        self.trace_decay = 0.9
        
        # History for analysis (memory-efficient ring buffer)
        self.history_size = 10000
        self.register_buffer('rpe_history', torch.zeros(self.history_size))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
    
    def forward(
        self,
        current_state: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
        actual_reward: Optional[torch.Tensor] = None,
        done: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RPE given current and next states.
        
        Args:
            current_state: Current representation [batch, state_dim]
            next_state: Next state (if available)
            actual_reward: Actual reward received
            done: Episode termination flag
            
        Returns:
            rpe: Reward prediction error
            metadata: Additional signals
        """
        batch_size = current_state.shape[0]
        
        # Estimate value of current state
        current_value = self.value_network(current_state).squeeze(-1)
        
        # Predict expected reward
        expected_reward = self.reward_predictor(current_state).squeeze(-1)
        
        if next_state is not None:
            # TD target: R + γ * V(s')
            next_value = self.value_network(next_state).squeeze(-1)
            if done:
                next_value = torch.zeros_like(next_value)
                
            td_target = (actual_reward if actual_reward is not None else 0) + self.gamma * next_value
            
            # RPE = TD target - current value estimate
            rpe = td_target - current_value
        else:
            # If no next state, use reward prediction error
            if actual_reward is not None:
                rpe = actual_reward - expected_reward
            else:
                rpe = torch.zeros(batch_size, device=current_state.device)
        
        # Update eligibility trace
        self.eligibility_trace = self.trace_decay * self.eligibility_trace + current_value.mean()
        
        # Store in history
        if rpe.numel() > 0:
            ptr = self.history_ptr.item()
            self.rpe_history[ptr] = rpe.mean().detach()
            self.history_ptr = (self.history_ptr + 1) % self.history_size
        
        metadata = {
            'current_value': current_value.mean().item(),
            'expected_reward': expected_reward.mean().item(),
            'eligibility_trace': self.eligibility_trace.item(),
            'rpe_variance': rpe.var().item() if rpe.numel() > 1 else 0,
        }
        
        return rpe, metadata
    
    def get_learning_modulation(self, rpe: torch.Tensor) -> torch.Tensor:
        """
        Convert RPE to learning rate modulation.
        
        Biological Parallel:
            - Large positive RPE → Enhanced LTP (potentiation)
            - Large negative RPE → Enhanced LTD (depression)
            - Zero RPE → No learning (prediction was correct)
        """
        # Sigmoid-like modulation centered at 1.0
        modulation = 1.0 + torch.tanh(rpe)  # Range: [0, 2]
        return modulation


class DopamineCircuit(nn.Module):
    """
    Complete Dopamine Circuit - The Motivational Core
    
    Biological Parallel:
        The VTA-NAc-PFC dopamine circuit controls:
        1. Motivation and drive
        2. Reward processing
        3. Working memory gating
        4. Attention modulation
        
    Implementation:
        - Tonic dopamine: Baseline level reflecting general motivation
        - Phasic dopamine: Burst responses to unexpected events
        - Modulates learning, memory, and attention across modules
        
    Efficiency:
        - Sparse computation (only updates on significant events)
        - O(1) dopamine level updates
        - Minimal memory footprint
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        baseline_dopamine: float = 0.5,
        decay_rate: float = 0.95,
        burst_threshold: float = 0.3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.baseline_dopamine = baseline_dopamine
        self.decay_rate = decay_rate
        self.burst_threshold = burst_threshold
        
        # RPE computation module
        self.rpe_module = RewardPredictionError(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )
        
        # Novelty detector (drives exploration)
        # High novelty → increased dopamine
        self.novelty_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Dopamine state buffer
        self.register_buffer('tonic_level', torch.tensor(baseline_dopamine))
        self.register_buffer('phasic_level', torch.tensor(0.0))
        self.register_buffer('adaptation_rate', torch.tensor(0.01))
        
        # Synaptic weight for integrating RPE into dopamine
        self.rpe_to_dopamine = nn.Parameter(torch.tensor(0.5))
        
        # History tracking
        self.register_buffer('dopamine_history', torch.zeros(1000))
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        action_taken: Optional[torch.Tensor] = None,
    ) -> Tuple[DopamineState, Dict]:
        """
        Compute current dopamine state.
        
        Args:
            state: Current internal state representation
            next_state: Next state (if available)
            reward: Actual reward received
            action_taken: Action that was taken (for credit assignment)
            
        Returns:
            DopamineState: Current dopamine levels and signals
            metadata: Additional information for debugging/analysis
        """
        # Compute RPE
        rpe, rpe_meta = self.rpe_module(state, next_state, reward)
        
        # Detect novelty
        novelty = self.novelty_detector(state).squeeze(-1)
        
        # Update tonic dopamine (slow adaptation)
        # Moves toward baseline with modulation from recent RPE
        avg_rpe = rpe.mean().detach()
        self.tonic_level = (
            self.decay_rate * self.tonic_level + 
            (1 - self.decay_rate) * (self.baseline_dopamine + 0.1 * avg_rpe)
        ).clamp(0.1, 0.9)
        
        # Compute phasic burst
        # Large positive RPE or high novelty → dopamine burst
        burst_signal = self.rpe_to_dopamine * avg_rpe + 0.3 * novelty.mean()
        
        if abs(burst_signal) > self.burst_threshold:
            self.phasic_level = burst_signal.clamp(-1, 1)
        else:
            # Decay phasic toward zero
            self.phasic_level = self.phasic_level * 0.8
        
        # Total dopamine level
        total_dopamine = (self.tonic_level + self.phasic_level).clamp(0, 1)
        
        # Record history
        idx = self.history_idx.item()
        self.dopamine_history[idx] = total_dopamine.detach()
        self.history_idx = (self.history_idx + 1) % 1000
        
        # Create state object
        da_state = DopamineState(
            level=total_dopamine.item(),
            burst=self.phasic_level.item(),
            rpe=avg_rpe.item(),
            expected_reward=rpe_meta['expected_reward'],
        )
        
        metadata = {
            'tonic_dopamine': self.tonic_level.item(),
            'phasic_dopamine': self.phasic_level.item(),
            'novelty': novelty.mean().item(),
            'total_dopamine': total_dopamine.item(),
            **rpe_meta,
        }
        
        return da_state, metadata
    
    def get_learning_modulation(self) -> float:
        """
        Get current learning rate modulation based on dopamine.
        
        Returns:
            Multiplier for learning rate [0.5, 2.0]
        """
        # Higher dopamine → faster learning
        modulation = 0.5 + 1.5 * self.tonic_level.item()
        
        # Phasic bursts temporarily boost learning
        if self.phasic_level.item() > 0:
            modulation *= (1 + 0.5 * self.phasic_level.item())
        
        return min(modulation, 2.0)
    
    def get_attention_modulation(self) -> float:
        """
        Get attention focus modulation.
        
        Returns:
            Multiplier for attention weights [0.5, 1.5]
        """
        # Higher dopamine → more focused attention
        return 0.5 + self.tonic_level.item()
    
    def reset(self):
        """Reset dopamine to baseline state."""
        self.tonic_level.fill_(self.baseline_dopamine)
        self.phasic_level.zero_()
        self.dopamine_history.zero_()
        self.history_idx.zero_()


class NeuromodulatorMix(nn.Module):
    """
    Extended Neuromodulator System
    
    Beyond dopamine, the brain uses multiple neuromodulators:
        - Dopamine: Reward, motivation, learning
        - Norepinephrine: Arousal, attention, stress response
        - Serotonin: Mood, patience, long-term planning
        - Acetylcholine: Memory encoding, attention
        
    This module provides a simplified simulation for future extension.
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        # Dopamine (primary, fully implemented above)
        self.register_buffer('dopamine', torch.tensor(0.5))
        
        # Norepinephrine (simplified: arousal/alertness)
        self.register_buffer('norepinephrine', torch.tensor(0.5))
        
        # Serotonin (simplified: patience/inhibition)
        self.register_buffer('serotonin', torch.tensor(0.5))
        
        # Acetylcholine (simplified: memory encoding strength)
        self.register_buffer('acetylcholine', torch.tensor(0.5))
        
        # Modulators affect global processing
        self.modulator_weights = nn.Parameter(torch.ones(4) / 4)
    
    def get_composite_modulation(self) -> Dict[str, float]:
        """Get modulation effects from all neuromodulators."""
        return {
            'learning_rate': 0.5 + self.dopamine.item(),
            'attention_focus': 0.5 + 0.5 * self.norepinephrine.item(),
            'patience': 0.5 + 0.5 * self.serotonin.item(),
            'memory_encoding': 0.5 + 0.5 * self.acetylcholine.item(),
        }
