"""
GENESIS Neuromodulation - Multi-Transmitter System

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

The brain uses multiple neuromodulators to control learning and behavior:

1. DOPAMINE (DA) - Reward & Motivation
   - Signals reward prediction error
   - High DA → "This is good, learn it!"
   
2. NOREPINEPHRINE (NE) - Arousal & Surprise
   - Signals unexpected events
   - High NE → "Pay attention!"
   
3. SEROTONIN (5-HT) - Mood & Patience
   - Modulates risk aversion
   - Low 5-HT → Impulsive, high → Patient
   
4. ACETYLCHOLINE (ACh) - Learning Gate
   - Gates synaptic plasticity
   - High ACh → "Learning mode ON"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NeuromodulatorState:
    """Current state of neuromodulatory system."""
    dopamine: float = 0.5
    norepinephrine: float = 0.5
    serotonin: float = 0.5
    acetylcholine: float = 0.5
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'dopamine': self.dopamine,
            'norepinephrine': self.norepinephrine,
            'serotonin': self.serotonin,
            'acetylcholine': self.acetylcholine,
        }
    
    def to_tensor(self, device: torch.device = None) -> torch.Tensor:
        t = torch.tensor([self.dopamine, self.norepinephrine, 
                         self.serotonin, self.acetylcholine])
        if device:
            t = t.to(device)
        return t


class Neuromodulator(nn.Module):
    """Base class for a single neuromodulator."""
    
    def __init__(
        self,
        name: str,
        baseline: float = 0.5,
        decay_rate: float = 0.1,
        min_level: float = 0.0,
        max_level: float = 1.0,
    ):
        super().__init__()
        
        self.name = name
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.min_level = min_level
        self.max_level = max_level
        
        self.register_buffer('level', torch.tensor(baseline))
        self.register_buffer('history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
    
    def update(self, stimulus: float):
        """Update level based on stimulus."""
        # Decay toward baseline
        self.level = self.level + self.decay_rate * (self.baseline - self.level)
        
        # Add stimulus
        self.level = self.level + stimulus
        
        # Clamp
        self.level = self.level.clamp(self.min_level, self.max_level)
        
        # Record history
        ptr = self.history_ptr.item()
        self.history[ptr] = self.level
        self.history_ptr = (self.history_ptr + 1) % 100
    
    def get_level(self) -> float:
        return self.level.item()
    
    def reset(self):
        self.level.fill_(self.baseline)
        self.history.zero_()


class DopamineSystem(Neuromodulator):
    """
    Dopamine system with RPE dynamics.
    """
    
    def __init__(self, state_dim: int, gamma: float = 0.99):
        super().__init__('dopamine', baseline=0.5, decay_rate=0.05)
        
        self.gamma = gamma
        
        # Value estimator
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Running baseline
        self.register_buffer('reward_baseline', torch.tensor(0.0))
    
    def compute_rpe(
        self,
        state: torch.Tensor,
        reward: float,
        next_state: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute reward prediction error."""
        with torch.no_grad():
            v_current = self.value_net(state).item()
            
            if next_state is not None:
                v_next = self.value_net(next_state).item()
            else:
                v_next = 0.0
            
            # TD error
            rpe = reward + self.gamma * v_next - v_current
            
            # Update baseline
            alpha = 0.01
            self.reward_baseline = (1 - alpha) * self.reward_baseline + alpha * reward
            
            # Normalize
            rpe = rpe - self.reward_baseline.item()
        
        return rpe
    
    def update_from_rpe(self, rpe: float):
        """Update dopamine level from RPE."""
        # Positive RPE → DA burst
        # Negative RPE → DA dip
        self.update(0.5 * rpe)


class GenesisLimbic(nn.Module):
    """
    Complete limbic system with all neuromodulators.
    """
    
    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Neuromodulators
        self.dopamine = DopamineSystem(state_dim, gamma)
        self.norepinephrine = Neuromodulator('norepinephrine', baseline=0.5, decay_rate=0.15)
        self.serotonin = Neuromodulator('serotonin', baseline=0.5, decay_rate=0.02)
        self.acetylcholine = Neuromodulator('acetylcholine', baseline=0.5, decay_rate=0.1)
        
        # State representation
        self.state_encoder = nn.Linear(state_dim, 64)
        
        # Surprise detector
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_var', torch.ones(state_dim))
    
    def process(
        self,
        state: torch.Tensor,
        reward: Optional[float] = None,
        next_state: Optional[torch.Tensor] = None,
        attention_signal: Optional[float] = None,
    ) -> NeuromodulatorState:
        """
        Process state and update all neuromodulators.
        
        Args:
            state: Current state representation
            reward: Optional reward signal
            next_state: Optional next state (for TD)
            attention_signal: Optional attention/salience signal
            
        Returns:
            Current neuromodulator state
        """
        # ═══════════════════════════════════════════════════════════════
        # 1. DOPAMINE (Reward Prediction Error)
        # ═══════════════════════════════════════════════════════════════
        if reward is not None:
            rpe = self.dopamine.compute_rpe(state, reward, next_state)
            self.dopamine.update_from_rpe(rpe)
        else:
            self.dopamine.update(0)  # Just decay
        
        # ═══════════════════════════════════════════════════════════════
        # 2. NOREPINEPHRINE (Surprise/Novelty)
        # ═══════════════════════════════════════════════════════════════
        # Compute surprise as deviation from running mean
        with torch.no_grad():
            if state.dim() > 1:
                state_mean = state.mean(0)
            else:
                state_mean = state
            
            # Update running stats
            alpha = 0.01
            self.state_mean = (1 - alpha) * self.state_mean + alpha * state_mean
            deviation = state_mean - self.state_mean
            self.state_var = (1 - alpha) * self.state_var + alpha * deviation ** 2
            
            # Surprise = magnitude of deviation
            surprise = (deviation.abs() / (self.state_var.sqrt() + 1e-8)).mean()
            surprise = torch.tanh(surprise).item()
        
        self.norepinephrine.update(surprise * 0.3)
        
        # ═══════════════════════════════════════════════════════════════
        # 3. ACETYLCHOLINE (Attention/Learning Gate)
        # ═══════════════════════════════════════════════════════════════
        if attention_signal is not None:
            self.acetylcholine.update(attention_signal * 0.5)
        else:
            # Use surprise as proxy for attention
            self.acetylcholine.update(surprise * 0.2)
        
        # ═══════════════════════════════════════════════════════════════
        # 4. SEROTONIN (Mood - slow dynamics)
        # ═══════════════════════════════════════════════════════════════
        # Serotonin increases with consistent positive reward
        if reward is not None:
            self.serotonin.update(reward * 0.05)
        else:
            self.serotonin.update(0)
        
        return self.get_state()
    
    def get_state(self) -> NeuromodulatorState:
        """Get current neuromodulator state."""
        return NeuromodulatorState(
            dopamine=self.dopamine.get_level(),
            norepinephrine=self.norepinephrine.get_level(),
            serotonin=self.serotonin.get_level(),
            acetylcholine=self.acetylcholine.get_level(),
        )
    
    def get_learning_modulation(self) -> Dict[str, float]:
        """
        Get modulation factors for learning.
        
        Returns:
            Dict with modulation factors
        """
        state = self.get_state()
        
        return {
            # DA gates reward-based learning
            'reward_learning': state.dopamine,
            
            # ACh gates all learning
            'learning_gate': max(0, state.acetylcholine - 0.3) / 0.7,
            
            # NE enhances learning for surprising events
            'surprise_boost': 1.0 + 0.5 * state.norepinephrine,
            
            # 5-HT affects exploration/exploitation
            'exploitation': state.serotonin,
            'exploration': 1.0 - state.serotonin,
        }
    
    def reset(self):
        """Reset all neuromodulators."""
        self.dopamine.reset()
        self.norepinephrine.reset()
        self.serotonin.reset()
        self.acetylcholine.reset()
