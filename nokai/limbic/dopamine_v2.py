"""
Dopamine Circuit V2 - Homeostatic Reward Prediction System

=============================================================================
BIOLOGICAL BACKGROUND - Why V1 Was Wrong
=============================================================================

The original DopamineCircuit had a critical flaw: dopamine could stay at 1.0
indefinitely, creating a "drugued brain" that couldn't differentiate between
success and failure.

In the real VTA (Ventral Tegmental Area):
    1. Dopamine responds to SURPRISE, not raw success
    2. The brain HABITUATES to constant stimulation (hedonic adaptation)
    3. Dopamine dips BELOW baseline when expected rewards are missing

Mathematical Foundation - Reward Prediction Error (RPE):
    
    δ(t) = R(t) + γ·V(s_{t+1}) - V(s_t)
    
    Where:
        - δ(t) = RPE at time t (the SURPRISE signal)
        - R(t) = actual reward received
        - V(s) = predicted value of state s
        - γ = discount factor (typically 0.99)
        
    Key insight: δ approaches 0 when the brain accurately predicts rewards!
    This is why rats stop getting dopamine hits from expected food pellets.

Homeostasis Implementation:
    
    We add an exponential moving average (EMA) baseline that adapts:
    
    B(t) = α·B(t-1) + (1-α)·DA(t-1)
    
    Then the effective dopamine signal becomes:
    
    DA_effective(t) = DA_raw(t) - B(t) + 0.5
    
    This ensures that constant stimulation leads to adaptation,
    while genuine surprises still produce strong signals.

=============================================================================

Author: Nōkai Neuro-Engineering Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class DopamineStateV2:
    """
    Enhanced dopamine state with homeostatic tracking.
    
    Biological Mapping:
        - tonic_level: Baseline motivation (VTA tonic firing ~4-5Hz)
        - phasic_burst: Reward-related spike (+10-20Hz above tonic)
        - phasic_dip: Omission response (-2-4Hz below tonic)
        - rpe: Reward Prediction Error - the LEARNING signal
        - surprise: Absolute surprise magnitude (|RPE|)
        - habituation: Current adaptation level (0=fresh, 1=fully habituated)
        - effective_signal: What the rest of the brain actually "feels"
    """
    tonic_level: float = 0.5
    phasic_burst: float = 0.0
    phasic_dip: float = 0.0
    rpe: float = 0.0
    surprise: float = 0.0
    habituation: float = 0.0
    effective_signal: float = 0.5
    
    @property
    def level(self) -> float:
        """Backwards compatibility with V1."""
        return self.effective_signal
    
    @property
    def burst(self) -> float:
        """Backwards compatibility with V1."""
        return self.phasic_burst - self.phasic_dip


# ============================================
# VALUE NETWORK (CRITIC)
# ============================================

class ValueNetwork(nn.Module):
    """
    State Value Estimator V(s) - The "Prediction" in Reward Prediction Error.
    
    Biological Parallel:
        The ventral striatum and orbitofrontal cortex maintain internal
        predictions of expected rewards. This network learns those predictions.
    
    Architecture:
        Simple MLP with LayerNorm for stability.
        Output is unbounded (rewards can be any scale).
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize final layer to output near-zero values
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of state.
        
        Args:
            state: State representation [batch, state_dim]
            
        Returns:
            value: Estimated value [batch]
        """
        return self.network(state).squeeze(-1)


# ============================================
# NOVELTY DETECTOR
# ============================================

class NoveltyDetector(nn.Module):
    """
    Detects novel/surprising inputs based on reconstruction error.
    
    Biological Parallel:
        The hippocampus and anterior cingulate cortex detect mismatches
        between expected and actual inputs, driving exploration.
    
    Implementation:
        Uses an autoencoder - reconstruction error indicates novelty.
        Novel inputs increase dopamine independently of reward.
    """
    
    def __init__(self, state_dim: int, latent_dim: int = 64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, state_dim),
        )
        
        # Running statistics for normalization
        self.register_buffer('error_mean', torch.tensor(0.1))
        self.register_buffer('error_std', torch.tensor(0.1))
        self.register_buffer('num_samples', torch.tensor(0, dtype=torch.long))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute novelty score.
        
        Args:
            state: Input state [batch, state_dim]
            
        Returns:
            novelty: Normalized novelty score [batch] in [0, 1]
            reconstruction_loss: Raw MSE loss for training
        """
        # Encode-decode
        latent = self.encoder(state)
        reconstructed = self.decoder(latent)
        
        # Reconstruction error per sample
        mse = ((state - reconstructed) ** 2).mean(dim=-1)
        
        # Update running statistics
        if self.training:
            batch_mean = mse.mean().detach()
            batch_std = mse.std().detach() + 1e-6
            
            # Exponential moving average
            alpha = 0.01
            self.error_mean = (1 - alpha) * self.error_mean + alpha * batch_mean
            self.error_std = (1 - alpha) * self.error_std + alpha * batch_std
            self.num_samples += state.shape[0]
        
        # Normalize to [0, 1] using z-score transformed through sigmoid
        z_score = (mse - self.error_mean) / (self.error_std + 1e-6)
        novelty = torch.sigmoid(z_score)  # High error = high novelty
        
        return novelty, mse


# ============================================
# HOMEOSTATIC BASELINE
# ============================================

class HomeostaticBaseline(nn.Module):
    """
    Implements hedonic adaptation through exponential moving average.
    
    Biological Parallel:
        The brain's reward system adapts to consistent stimulation:
        - Chronic drug use leads to tolerance (need more for same effect)
        - Lottery winners return to baseline happiness within months
        - Neurons reduce firing rate to constant stimulation
    
    Mathematical Model:
        baseline(t) = α * baseline(t-1) + (1-α) * signal(t-1)
        
        With α = 0.99 (slow adaptation) for tonic
        And α = 0.9 (fast adaptation) for phasic bursts
    
    The effective signal becomes:
        effective(t) = raw(t) - baseline(t) + set_point
        
    Where set_point = 0.5 (neutral dopamine level).
    """
    
    def __init__(
        self,
        tonic_adaptation_rate: float = 0.01,   # α for tonic (slow)
        phasic_adaptation_rate: float = 0.1,   # α for phasic (faster)
        set_point: float = 0.5,                 # Target baseline
        min_sensitivity: float = 0.2,           # Never fully habituate
    ):
        super().__init__()
        
        # Adaptation rates (higher = faster adaptation)
        self.tonic_alpha = tonic_adaptation_rate
        self.phasic_alpha = phasic_adaptation_rate
        self.set_point = set_point
        self.min_sensitivity = min_sensitivity
        
        # Running baselines
        self.register_buffer('tonic_baseline', torch.tensor(set_point))
        self.register_buffer('phasic_baseline', torch.tensor(0.0))
        
        # Track how long signal has been constant (for habituation measure)
        self.register_buffer('stability_counter', torch.tensor(0, dtype=torch.long))
        self.register_buffer('last_signal', torch.tensor(set_point))
        
        # Track total habituation
        self.register_buffer('habituation_level', torch.tensor(0.0))
    
    def forward(
        self,
        raw_tonic: torch.Tensor,
        raw_phasic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Apply homeostatic adjustment.
        
        Args:
            raw_tonic: Raw tonic dopamine level [batch] or scalar
            raw_phasic: Raw phasic burst/dip [batch] or scalar
            
        Returns:
            adjusted_tonic: Homeostasis-adjusted tonic level
            adjusted_phasic: Homeostasis-adjusted phasic response
            metadata: Habituation statistics
        """
        # Get scalar values for baseline update
        tonic_mean = raw_tonic.mean() if raw_tonic.dim() > 0 else raw_tonic
        phasic_mean = raw_phasic.mean() if raw_phasic.dim() > 0 else raw_phasic
        
        # Update tonic baseline (slow adaptation)
        self.tonic_baseline = (
            (1 - self.tonic_alpha) * self.tonic_baseline + 
            self.tonic_alpha * tonic_mean.detach()
        )
        
        # Update phasic baseline (faster adaptation for bursts)
        self.phasic_baseline = (
            (1 - self.phasic_alpha) * self.phasic_baseline +
            self.phasic_alpha * phasic_mean.detach()
        )
        
        # Compute deviation from baseline
        tonic_deviation = raw_tonic - self.tonic_baseline
        phasic_deviation = raw_phasic - self.phasic_baseline
        
        # Compute sensitivity (inverse of habituation)
        # As habituation increases, sensitivity decreases but never below min
        sensitivity = max(
            self.min_sensitivity,
            1.0 - self.habituation_level.item() * (1 - self.min_sensitivity)
        )
        
        # Apply sensitivity scaling
        adjusted_tonic = self.set_point + tonic_deviation * sensitivity
        adjusted_phasic = phasic_deviation * sensitivity
        
        # Update habituation tracking
        signal_change = abs(tonic_mean - self.last_signal).item()
        if signal_change < 0.01:  # Signal is stable
            self.stability_counter = self.stability_counter + 1
            # Increase habituation logarithmically
            self.habituation_level = torch.tanh(
                self.stability_counter.float() / 100.0
            )
        else:  # Signal changed - reduce habituation
            self.stability_counter = torch.clamp(self.stability_counter - 10, min=0)
            self.habituation_level = self.habituation_level * 0.9
        
        self.last_signal = tonic_mean.detach().clone()
        
        metadata = {
            'tonic_baseline': self.tonic_baseline.item(),
            'phasic_baseline': self.phasic_baseline.item(),
            'sensitivity': sensitivity,
            'habituation': self.habituation_level.item(),
            'stability_steps': self.stability_counter.item(),
        }
        
        return adjusted_tonic, adjusted_phasic, metadata
    
    def reset(self):
        """Reset to fresh (no habituation) state."""
        self.tonic_baseline.fill_(self.set_point)
        self.phasic_baseline.zero_()
        self.stability_counter.zero_()
        self.habituation_level.zero_()
        self.last_signal.fill_(self.set_point)


# ============================================
# MAIN DOPAMINE CIRCUIT V2
# ============================================

class DopamineCircuitV2(nn.Module):
    """
    Homeostatic Dopamine Circuit - Reward Prediction Error with Adaptation.
    
    ==========================================================================
    BIOLOGICAL PARALLEL: THE VTA-STRIATUM DOPAMINE PATHWAY
    ==========================================================================
    
    Dopaminergic neurons in the Ventral Tegmental Area (VTA) encode:
    
    1. REWARD PREDICTION ERROR (RPE):
       - Fires ABOVE baseline for unexpected rewards (positive surprise)
       - Fires AT baseline for expected rewards (no learning needed)
       - Fires BELOW baseline for omitted expected rewards (negative surprise)
       
    2. NOVELTY/SALIENCE:
       - Novel stimuli increase dopamine independently of reward
       - This drives exploration and attention
       
    3. HOMEOSTASIS:
       - Constant stimulation leads to receptor downregulation
       - The brain ADAPTS to prevent permanent over/under-activation
    
    ==========================================================================
    MATHEMATICAL FORMULATION
    ==========================================================================
    
    Let R(t) be the reward at time t, V(s) the value function, and:
    
    1. Reward Prediction Error:
       δ(t) = R(t) + γ·V(s_{t+1}) - V(s_t)
       
    2. Phasic response:
       burst(t) = max(0, k₁·δ(t))      if δ(t) > 0 (positive surprise)
       dip(t) = max(0, k₂·|δ(t)|)      if δ(t) < 0 (negative surprise)
       
    3. Tonic level with homeostasis:
       tonic_raw(t) = baseline + μ·avg_reward + ν·novelty
       tonic(t) = homeostasis(tonic_raw(t))
       
    4. Effective signal (what the brain uses):
       DA(t) = tonic(t) + burst(t) - dip(t)
       
    Where homeostasis() applies the adaptive baseline adjustment.
    
    ==========================================================================
    USAGE
    ==========================================================================
    
        circuit = DopamineCircuitV2(state_dim=256)
        
        # During training step:
        state, metadata = circuit(
            state=hidden_representation,  # [batch, dim]
            reward=loss_improvement,       # [batch] or scalar
            next_state=next_hidden,        # Optional
        )
        
        # Use for learning rate modulation:
        lr_multiplier = circuit.get_learning_modulation()
        
        # state.level is now properly adaptive!
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        baseline_dopamine: float = 0.5,
        # Homeostasis parameters
        tonic_adaptation_rate: float = 0.01,
        phasic_adaptation_rate: float = 0.1,
        # Burst/dip response curves
        burst_gain: float = 1.0,
        dip_gain: float = 0.5,
        burst_threshold: float = 0.1,
        # Novelty contribution
        novelty_weight: float = 0.3,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.baseline_dopamine = baseline_dopamine
        self.burst_gain = burst_gain
        self.dip_gain = dip_gain
        self.burst_threshold = burst_threshold
        self.novelty_weight = novelty_weight
        
        # =====================================
        # SUBMODULES
        # =====================================
        
        # Value function for RPE computation
        self.value_network = ValueNetwork(state_dim, hidden_dim)
        
        # Novelty detector
        self.novelty_detector = NoveltyDetector(state_dim, latent_dim=64)
        
        # Homeostatic baseline adjustment
        self.homeostasis = HomeostaticBaseline(
            tonic_adaptation_rate=tonic_adaptation_rate,
            phasic_adaptation_rate=phasic_adaptation_rate,
            set_point=baseline_dopamine,
        )
        
        # =====================================
        # STATE BUFFERS
        # =====================================
        
        # Current dopamine levels (accessible between forward passes)
        self.register_buffer('current_tonic', torch.tensor(baseline_dopamine))
        self.register_buffer('current_phasic', torch.tensor(0.0))
        self.register_buffer('current_rpe', torch.tensor(0.0))
        self.register_buffer('current_novelty', torch.tensor(0.0))
        
        # History for analysis
        self.register_buffer('rpe_history', torch.zeros(1000))
        self.register_buffer('dopamine_history', torch.zeros(1000))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        
        # Running average of rewards (for baseline reward expectation)
        self.register_buffer('avg_reward', torch.tensor(0.0))
        self.reward_ema_alpha = 0.01
    
    def compute_rpe(
        self,
        state: torch.Tensor,
        next_state: Optional[torch.Tensor],
        reward: Optional[torch.Tensor],
        done: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Reward Prediction Error.
        
        The core formula:
            δ = R + γ·V(s') - V(s)
            
        For language models without explicit RL:
            R = -loss_improvement (lower loss = positive reward)
            s = current hidden state  
            s' = next hidden state (or None for single-step)
        
        Args:
            state: Current state [batch, state_dim]
            next_state: Next state [batch, state_dim] or None
            reward: Reward signal [batch] or scalar or None
            done: Episode termination flag
            
        Returns:
            rpe: Reward prediction error [batch]
            metadata: Value estimates and other info
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Current value estimate
        current_value = self.value_network(state)
        
        # Handle reward
        if reward is None:
            reward = torch.zeros(batch_size, device=device)
        elif reward.dim() == 0:
            reward = reward.expand(batch_size)
        
        # Update running reward average
        self.avg_reward = (
            (1 - self.reward_ema_alpha) * self.avg_reward +
            self.reward_ema_alpha * reward.mean().detach()
        )
        
        # Compute TD target
        if next_state is not None:
            next_value = self.value_network(next_state)
            if done:
                next_value = torch.zeros_like(next_value)
            td_target = reward + self.gamma * next_value
        else:
            # No next state - use reward directly as target
            td_target = reward
        
        # RPE = TD target - current estimate
        rpe = td_target - current_value
        
        metadata = {
            'current_value': current_value.mean().item(),
            'td_target': td_target.mean().item() if td_target is not None else 0,
            'avg_reward': self.avg_reward.item(),
        }
        
        return rpe, metadata
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        action_taken: Optional[torch.Tensor] = None,  # For compatibility
    ) -> Tuple[DopamineStateV2, Dict]:
        """
        Compute homeostatic dopamine response.
        
        Args:
            state: Current internal state [batch, state_dim]
            next_state: Next state (if available)
            reward: External reward signal
            action_taken: (Unused, for API compatibility)
            
        Returns:
            DopamineStateV2: Complete dopamine state with homeostasis
            metadata: Detailed statistics for logging/debugging
        """
        batch_size = state.shape[0]
        device = state.device
        
        # =====================================
        # 1. COMPUTE REWARD PREDICTION ERROR
        # =====================================
        rpe, rpe_meta = self.compute_rpe(state, next_state, reward)
        rpe_mean = rpe.mean()
        
        # =====================================
        # 2. DETECT NOVELTY
        # =====================================
        novelty, reconstruction_loss = self.novelty_detector(state)
        novelty_mean = novelty.mean()
        
        # =====================================
        # 3. COMPUTE RAW PHASIC RESPONSE
        # =====================================
        
        # Positive surprise (unexpected reward) -> burst
        positive_rpe = F.relu(rpe_mean - self.burst_threshold)
        phasic_burst = self.burst_gain * positive_rpe
        
        # Negative surprise (expected reward missing) -> dip
        negative_rpe = F.relu(-rpe_mean - self.burst_threshold)
        phasic_dip = self.dip_gain * negative_rpe
        
        # Combined phasic
        raw_phasic = phasic_burst - phasic_dip
        
        # =====================================
        # 4. COMPUTE RAW TONIC LEVEL
        # =====================================
        
        # Tonic influenced by reward average and novelty
        raw_tonic = (
            self.baseline_dopamine +
            0.1 * self.avg_reward +  # Longer-term reward history
            self.novelty_weight * novelty_mean  # Exploration drive
        )
        raw_tonic = raw_tonic.clamp(0.1, 0.9)
        
        # =====================================
        # 5. APPLY HOMEOSTASIS
        # =====================================
        
        adjusted_tonic, adjusted_phasic, homeo_meta = self.homeostasis(
            raw_tonic, raw_phasic
        )
        
        # =====================================
        # 6. COMPUTE EFFECTIVE SIGNAL
        # =====================================
        
        # The signal that the rest of the brain actually uses
        effective = (adjusted_tonic + adjusted_phasic).clamp(0, 1)
        
        # Compute absolute surprise for learning modulation
        surprise = abs(rpe_mean.item())
        
        # =====================================
        # 7. UPDATE STATE BUFFERS
        # =====================================
        
        self.current_tonic = adjusted_tonic.detach()
        self.current_phasic = adjusted_phasic.detach()
        self.current_rpe = rpe_mean.detach()
        self.current_novelty = novelty_mean.detach()
        
        # Record history
        ptr = self.history_ptr.item()
        self.rpe_history[ptr] = rpe_mean.detach()
        self.dopamine_history[ptr] = effective.detach()
        self.history_ptr = (self.history_ptr + 1) % 1000
        
        # =====================================
        # 8. BUILD OUTPUT STATE
        # =====================================
        
        da_state = DopamineStateV2(
            tonic_level=self.current_tonic.item(),
            phasic_burst=phasic_burst.item(),
            phasic_dip=phasic_dip.item(),
            rpe=rpe_mean.item(),
            surprise=surprise,
            habituation=homeo_meta['habituation'],
            effective_signal=effective.item(),
        )
        
        metadata = {
            'tonic_dopamine': self.current_tonic.item(),
            'phasic_dopamine': adjusted_phasic.item(),
            'total_dopamine': effective.item(),
            'rpe': rpe_mean.item(),
            'surprise': surprise,
            'novelty': novelty_mean.item(),
            'reconstruction_loss': reconstruction_loss.mean().item(),
            **rpe_meta,
            **homeo_meta,
        }
        
        return da_state, metadata
    
    def get_learning_modulation(self) -> float:
        """
        Get learning rate modulation based on dopamine state.
        
        Biological Parallel:
            - High dopamine (surprise/novelty) -> Enhanced LTP
            - Low dopamine (no surprise) -> Reduced plasticity
            - Very low dopamine (negative surprise) -> Enhanced LTD
        
        Returns:
            Multiplier for learning rate [0.5, 2.0]
        """
        # Base modulation from effective dopamine level
        # DA=0.5 (baseline) -> mod=1.0
        # DA=1.0 (max) -> mod=2.0
        # DA=0.0 (min) -> mod=0.5
        tonic_mod = 0.5 + 1.5 * self.current_tonic.item()
        
        # Surprise bonus (absolute RPE boosts learning)
        surprise_bonus = min(0.5, abs(self.current_rpe.item()) * 0.5)
        
        # Novelty bonus
        novelty_bonus = min(0.3, self.current_novelty.item() * 0.3)
        
        # Combined modulation
        modulation = tonic_mod + surprise_bonus + novelty_bonus
        
        # Clamp to reasonable range
        return min(2.0, max(0.5, modulation))
    
    def get_attention_modulation(self) -> float:
        """
        Get attention focus modulation.
        
        Higher dopamine = more focused attention on current input.
        
        Returns:
            Multiplier for attention weights [0.5, 1.5]
        """
        return 0.5 + self.current_tonic.item()
    
    def reset(self):
        """Reset to baseline state (fresh brain)."""
        self.current_tonic.fill_(self.baseline_dopamine)
        self.current_phasic.zero_()
        self.current_rpe.zero_()
        self.current_novelty.zero_()
        self.avg_reward.zero_()
        
        self.rpe_history.zero_()
        self.dopamine_history.zero_()
        self.history_ptr.zero_()
        
        self.homeostasis.reset()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dopamine system statistics."""
        # Get valid history entries
        ptr = self.history_ptr.item()
        if ptr > 0:
            valid_rpe = self.rpe_history[:ptr]
            valid_da = self.dopamine_history[:ptr]
        else:
            valid_rpe = self.rpe_history
            valid_da = self.dopamine_history
        
        return {
            'current_tonic': self.current_tonic.item(),
            'current_phasic': self.current_phasic.item(),
            'current_rpe': self.current_rpe.item(),
            'current_novelty': self.current_novelty.item(),
            'avg_reward': self.avg_reward.item(),
            'habituation': self.homeostasis.habituation_level.item(),
            'rpe_mean': valid_rpe.mean().item(),
            'rpe_std': valid_rpe.std().item(),
            'da_mean': valid_da.mean().item(),
            'da_std': valid_da.std().item(),
        }


# ============================================
# BACKWARDS COMPATIBILITY WRAPPER
# ============================================

# Alias for drop-in replacement
DopamineCircuit = DopamineCircuitV2
DopamineState = DopamineStateV2
