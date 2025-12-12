"""
GENESIS Learning - Local Learning Rules (No Backpropagation!)

=============================================================================
PHILOSOPHY
=============================================================================

The brain doesn't do backpropagation. It uses LOCAL learning rules where
each synapse has access only to:
1. Pre-synaptic activity (the input)
2. Post-synaptic activity (its own output)
3. Timing information (trace/eligibility)
4. Global neuromodulatory signals (reward, surprise)

This module implements biologically-plausible learning that achieves:
- One-shot learning (via STDP + strong eligibility)
- Continual learning (via homeostasis + EWC-like protection)
- Reward-guided learning (via RPE modulation)

=============================================================================
LEARNING RULES
=============================================================================

1. STDP (Spike-Timing Dependent Plasticity):
   Δw = A+ · e_pre · post - A- · e_post · pre
   
2. RPE (Reward Prediction Error):
   δ = r + γ·V(s') - V(s)
   Δw = δ · eligibility · sign(δ)
   
3. Homeostatic:
   Δw = λ · (ρ_target - ρ_actual) · w
   
4. EWC-like protection:
   Penalty = Σ F_i · (w_i - w*_i)²

Combined:
   Δw = ACh · [STDP + DA·RPE + Homeo] - EWC_penalty

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class GenesisLearningConfig:
    """Configuration for GENESIS local learning."""
    
    # STDP
    a_plus: float = 0.01           # LTP amplitude
    a_minus: float = 0.005         # LTD amplitude
    tau_plus: float = 20.0         # LTP time constant (ms)
    tau_minus: float = 20.0        # LTD time constant (ms)
    
    # RPE
    gamma: float = 0.99            # Discount factor
    rpe_scale: float = 0.1         # RPE learning rate
    
    # Homeostasis
    target_firing_rate: float = 0.05  # Target sparsity
    homeostatic_rate: float = 0.001    # Homeostatic learning rate
    
    # EWC (Elastic Weight Consolidation)
    ewc_lambda: float = 100.0      # EWC penalty strength
    fisher_update_rate: float = 0.01  # Fisher info update rate
    
    # Neuromodulation gates
    min_acetylcholine: float = 0.3  # Min ACh for learning
    dopamine_scale: float = 2.0     # DA multiplier for RPE
    
    # Stability
    weight_clip: float = 5.0        # Max weight magnitude
    max_update: float = 0.1         # Max update magnitude


class STDPRule(nn.Module):
    """
    Spike-Timing Dependent Plasticity implementation.
    
    If pre fires before post → LTP (strengthen)
    If post fires before pre → LTD (weaken)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        a_plus: float = 0.01,
        a_minus: float = 0.005,
        tau: float = 20.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau = tau
        
        # Eligibility traces
        self.register_buffer('pre_trace', torch.zeros(in_features))
        self.register_buffer('post_trace', torch.zeros(out_features))
    
    def update_traces(self, pre: torch.Tensor, post: torch.Tensor, dt: float = 1.0):
        """Update eligibility traces with new activity."""
        decay = math.exp(-dt / self.tau)
        
        # Average over batch if needed
        if pre.dim() > 1:
            pre = pre.mean(0)
        if post.dim() > 1:
            post = post.mean(0)
        
        self.pre_trace = decay * self.pre_trace + (1 - decay) * pre.abs().detach()
        self.post_trace = decay * self.post_trace + (1 - decay) * post.abs().detach()
    
    def compute_update(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Compute STDP weight update.
        
        Returns:
            delta_w [out_features, in_features]
        """
        # Average over batch
        if pre.dim() > 1:
            pre = pre.mean(0)
        if post.dim() > 1:
            post = post.mean(0)
        
        # LTP: pre trace × current post (pre came first)
        # Result: [out_features, in_features]
        ltp = self.a_plus * torch.outer(post, self.pre_trace)
        
        # LTD: post trace × current pre (post came first)  
        # Also [out_features, in_features]
        ltd = self.a_minus * torch.outer(self.post_trace, pre)
        
        delta = ltp - ltd
        
        # Update traces for next step
        self.update_traces(pre, post)
        
        return delta
    
    def reset(self):
        """Reset traces."""
        self.pre_trace.zero_()
        self.post_trace.zero_()


class RPEComputation(nn.Module):
    """
    Reward Prediction Error computation (TD-learning style).
    
    δ = r + γ·V(s') - V(s)
    
    Positive δ → unexpected reward → increase connection
    Negative δ → expected reward missing → decrease connection
    """
    
    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        
        self.gamma = gamma
        self.lr = learning_rate
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        
        # Running statistics for normalization
        self.register_buffer('rpe_mean', torch.tensor(0.0))
        self.register_buffer('rpe_std', torch.tensor(1.0))
        self.register_buffer('baseline', torch.tensor(0.0))
    
    def compute_value(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate value of state."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.value_net(state).squeeze(-1)
    
    def compute_rpe(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Reward Prediction Error.
        
        Returns:
            rpe: Normalized RPE scalar
            info: Dict with breakdown
        """
        # Current and next value estimates
        v_current = self.compute_value(state)
        v_next = self.compute_value(next_state)
        
        # TD error
        td_target = reward + self.gamma * v_next
        rpe = td_target - v_current
        
        # Update running stats
        with torch.no_grad():
            alpha = 0.01
            self.rpe_mean = (1 - alpha) * self.rpe_mean + alpha * rpe.mean()
            self.rpe_std = (1 - alpha) * self.rpe_std + alpha * rpe.std().clamp(min=0.1)
            self.baseline = (1 - alpha) * self.baseline + alpha * v_current.mean()
        
        # Normalize
        rpe_normalized = (rpe - self.rpe_mean) / (self.rpe_std + 1e-8)
        
        info = {
            'rpe_raw': rpe.mean().item(),
            'rpe_normalized': rpe_normalized.mean().item(),
            'value_current': v_current.mean().item(),
            'value_next': v_next.mean().item(),
            'baseline': self.baseline.item(),
        }
        
        return rpe_normalized, info
    
    def update_value_net(self, state: torch.Tensor, target: torch.Tensor):
        """Update value network towards target (for value learning)."""
        v = self.compute_value(state)
        loss = F.mse_loss(v, target)
        
        # Manual gradient update (local!)
        loss.backward()


class HomeostaticRegulator(nn.Module):
    """
    Homeostatic plasticity to maintain stable firing rates.
    
    If firing rate too high → decrease weights
    If firing rate too low → increase weights
    """
    
    def __init__(
        self,
        num_neurons: int,
        target_rate: float = 0.05,
        time_constant: float = 1000.0,
    ):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.target_rate = target_rate
        self.tau = time_constant
        
        # Running firing rate estimate
        self.register_buffer('firing_rate', torch.full((num_neurons,), target_rate))
    
    def update_rate(self, activity: torch.Tensor):
        """Update firing rate estimate."""
        if activity.dim() > 1:
            activity = activity.mean(0)
        
        alpha = 1.0 / self.tau
        current_rate = activity.abs().clamp(0, 1)
        self.firing_rate = (1 - alpha) * self.firing_rate + alpha * current_rate
    
    def compute_scaling(self) -> torch.Tensor:
        """
        Compute multiplicative scaling factor for homeostasis.
        
        Returns:
            scale [num_neurons]: >1 if too quiet, <1 if too active
        """
        # Ratio: target / actual
        ratio = self.target_rate / (self.firing_rate + 1e-8)
        
        # Soft scaling (not too aggressive)
        scale = torch.pow(ratio, 0.1)  # Gentle adjustment
        
        return scale.clamp(0.9, 1.1)


class ElasticWeightConsolidation:
    """
    EWC-like mechanism to protect important weights.
    
    Computes Fisher Information to identify important weights,
    then penalizes changes to those weights.
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 100.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store anchor weights and Fisher info
        self.anchor_weights = {}
        self.fisher_info = {}
        
        # Initialize
        self._init_from_model()
    
    def _init_from_model(self):
        """Initialize storage from model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.anchor_weights[name] = param.data.clone()
                self.fisher_info[name] = torch.zeros_like(param)
    
    def update_fisher(self, gradients: Dict[str, torch.Tensor], rate: float = 0.01):
        """
        Update Fisher Information from gradients.
        
        Fisher ≈ E[∇log p(y|x;θ)²] ≈ gradient magnitude
        """
        for name, grad in gradients.items():
            if name in self.fisher_info:
                self.fisher_info[name] = (
                    (1 - rate) * self.fisher_info[name] +
                    rate * grad.pow(2)
                )
    
    def consolidate(self):
        """
        Update anchor weights to current weights.
        
        Call this after completing a task to "lock in" important weights.
        """
        for name, param in self.model.named_parameters():
            if name in self.anchor_weights:
                self.anchor_weights[name] = param.data.clone()
    
    def compute_penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty for current weights.
        
        Penalty = Σ F_i * (θ_i - θ*_i)²
        """
        total_penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                fisher = self.fisher_info[name]
                anchor = self.anchor_weights[name]
                
                penalty = fisher * (param - anchor).pow(2)
                total_penalty = total_penalty + penalty.sum()
        
        return self.lambda_ewc * total_penalty


class GenesisLearning(nn.Module):
    """
    Complete GENESIS local learning system.
    
    Combines:
    - STDP for correlation-based learning
    - RPE for reward-guided learning
    - Homeostasis for stability
    - EWC for continual learning
    
    All gated by neuromodulatory signals.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[GenesisLearningConfig] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or GenesisLearningConfig()
        
        # Component learning rules
        self.stdp = STDPRule(
            in_features, out_features,
            a_plus=self.config.a_plus,
            a_minus=self.config.a_minus,
        )
        
        self.homeostasis = HomeostaticRegulator(
            out_features,
            target_rate=self.config.target_firing_rate,
        )
        
        # Statistics
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_delta', torch.tensor(0.0))
    
    def apply_local_update(
        self,
        weight: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        dopamine: float = 1.0,
        acetylcholine: float = 1.0,
        rpe: Optional[float] = None,
    ) -> Tuple[bool, Dict]:
        """
        Apply complete local learning update to weight matrix.
        
        Args:
            weight: Weight matrix [out, in] to update
            pre: Pre-synaptic activations
            post: Post-synaptic activations
            dopamine: DA level (scales RPE component)
            acetylcholine: ACh level (gates all learning)
            rpe: Optional RPE value (if None, RPE component skipped)
            
        Returns:
            success: Whether update was applied
            info: Dict with detailed statistics
        """
        cfg = self.config
        
        # ═══════════════════════════════════════════════════════════════
        # GATE: Check acetylcholine (attention/learning gate)
        # ═══════════════════════════════════════════════════════════════
        if acetylcholine < cfg.min_acetylcholine:
            return False, {'reason': 'ACh below threshold'}
        
        # ═══════════════════════════════════════════════════════════════
        # 1. STDP COMPONENT
        # ═══════════════════════════════════════════════════════════════
        delta_stdp = self.stdp.compute_update(pre, post)
        
        # ═══════════════════════════════════════════════════════════════
        # 2. RPE COMPONENT (if reward signal available)
        # ═══════════════════════════════════════════════════════════════
        delta_rpe = torch.zeros_like(weight)
        if rpe is not None:
            # Get eligibility trace: outer product of post_trace and pre_trace
            # Result should be [out_features, in_features] like weight
            trace = torch.outer(self.stdp.post_trace, self.stdp.pre_trace)
            
            # RPE modulated by dopamine
            effective_rpe = rpe * cfg.dopamine_scale * dopamine
            delta_rpe = cfg.rpe_scale * effective_rpe * trace
        
        # ═══════════════════════════════════════════════════════════════
        # 3. HOMEOSTATIC COMPONENT
        # ═══════════════════════════════════════════════════════════════
        self.homeostasis.update_rate(post)
        homeo_scale = self.homeostasis.compute_scaling()
        
        # Homeostatic adjustment: scale weights toward target firing rate
        delta_homeo = cfg.homeostatic_rate * (homeo_scale.unsqueeze(1) - 1) * weight
        
        # ═══════════════════════════════════════════════════════════════
        # 4. COMBINE WITH ACH GATING
        # ═══════════════════════════════════════════════════════════════
        ach_gate = (acetylcholine - cfg.min_acetylcholine) / (1 - cfg.min_acetylcholine)
        
        total_delta = ach_gate * (delta_stdp + delta_rpe + delta_homeo)
        
        # ═══════════════════════════════════════════════════════════════
        # 5. STABILITY: Clip update magnitude
        # ═══════════════════════════════════════════════════════════════
        total_delta = total_delta.clamp(-cfg.max_update, cfg.max_update)
        
        # ═══════════════════════════════════════════════════════════════
        # 6. APPLY UPDATE
        # ═══════════════════════════════════════════════════════════════
        with torch.no_grad():
            weight.data.add_(total_delta)
            
            # Clip weights for stability
            weight.data.clamp_(-cfg.weight_clip, cfg.weight_clip)
        
        # ═══════════════════════════════════════════════════════════════
        # 7. STATISTICS
        # ═══════════════════════════════════════════════════════════════
        self.update_count += 1
        self.total_delta = self.total_delta + total_delta.abs().mean()
        
        info = {
            'delta_stdp_mean': delta_stdp.abs().mean().item(),
            'delta_rpe_mean': delta_rpe.abs().mean().item(),
            'delta_homeo_mean': delta_homeo.abs().mean().item(),
            'total_delta_mean': total_delta.abs().mean().item(),
            'ach_gate': ach_gate,
            'dopamine': dopamine,
            'firing_rate': self.homeostasis.firing_rate.mean().item(),
            'update_count': self.update_count.item(),
        }
        
        return True, info
    
    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        count = max(1, self.update_count.item())
        return {
            'update_count': count,
            'avg_delta': (self.total_delta / count).item(),
            'mean_firing_rate': self.homeostasis.firing_rate.mean().item(),
            'firing_rate_std': self.homeostasis.firing_rate.std().item(),
        }
    
    def reset(self):
        """Reset learning state."""
        self.stdp.reset()
        self.update_count.zero_()
        self.total_delta.zero_()


class GenesisLearningLayer(nn.Module):
    """
    Linear layer with fully integrated GENESIS learning.
    
    Drop-in replacement for nn.Linear that learns locally!
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_ternary: bool = True,
        config: Optional[GenesisLearningConfig] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_ternary = use_ternary
        
        # Weight matrix
        if use_ternary:
            from nokai.genesis.ternary import TernaryLinear
            self.linear = TernaryLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Learning system
        self.learner = GenesisLearning(in_features, out_features, config)
        
        # Store activations
        self.last_pre = None
        self.last_post = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (stores activations for learning)."""
        output = self.linear(x)
        
        if self.training:
            self.last_pre = x.detach()
            self.last_post = output.detach()
        
        return output
    
    def local_learn(
        self,
        dopamine: float = 1.0,
        acetylcholine: float = 1.0,
        rpe: Optional[float] = None,
    ) -> Dict:
        """
        Apply local learning using stored activations.
        
        Call this DURING the forward pass, before loss.backward()!
        """
        if self.last_pre is None or self.last_post is None:
            return {'error': 'No activations stored'}
        
        # Get weight reference
        if self.use_ternary:
            weight = self.linear.weight
        else:
            weight = self.linear.weight
        
        success, info = self.learner.apply_local_update(
            weight=weight,
            pre=self.last_pre,
            post=self.last_post,
            dopamine=dopamine,
            acetylcholine=acetylcholine,
            rpe=rpe,
        )
        
        return info
