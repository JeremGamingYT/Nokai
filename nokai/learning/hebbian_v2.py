"""
Hebbian Learning V2 - Real-Time Synaptic Plasticity

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

"Neurons that fire together, wire together" - Donald Hebb, 1949

The human brain doesn't wait for backpropagation to learn. Synaptic 
plasticity happens IMMEDIATELY when neurons co-activate:

1. LONG-TERM POTENTIATION (LTP):
   - When presynaptic neuron A consistently activates postsynaptic neuron B
   - The synapse A→B becomes STRONGER
   - Mediated by NMDA receptor activation and calcium influx
   
2. LONG-TERM DEPRESSION (LTD):
   - When activation is out of sync or weak
   - The synapse becomes WEAKER
   - Prevents "runaway potentiation"

3. SPIKE-TIMING DEPENDENT PLASTICITY (STDP):
   - Timing matters! Pre before Post = LTP, Post before Pre = LTD
   - Creates causal learning: "A caused B"
   
4. METAPLASTICITY (Homeostatic):
   - The plasticity rules themselves adapt
   - Prevents excessive potentiation or depression
   - Implemented via sliding thresholds

=============================================================================
MATHEMATICAL FORMULATION
=============================================================================

Basic Hebbian Rule:
    Δw_ij = η · x_i · x_j
    
Where:
    - w_ij = synapse from neuron i to neuron j
    - η = learning rate
    - x_i, x_j = pre and post-synaptic activations

Oja's Rule (adds normalization to prevent unbounded growth):
    Δw_ij = η · (x_j · x_i - α · x_j² · w_ij)

BCM Rule (sliding threshold for metaplasticity):
    Δw_ij = η · x_i · x_j · (x_j - θ)
    
Where θ = E[x_j²] (average squared post-activation)

Our Implementation: Oja + BCM Hybrid with Dopamine Modulation:
    Δw_ij = η · DA · (x_j · (x_i - α·x_j·w_ij) · (x_j - θ))
    
This combines:
    - Hebbian correlation learning
    - Oja's weight decay for stability  
    - BCM's sliding threshold for metaplasticity
    - Dopamine gating for reward-relevant learning

=============================================================================

Author: Nōkai Neuro-Engineering Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import math


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class HebbianConfig:
    """
    Configuration for Hebbian learning.
    
    Biological Mapping:
        - learning_rate: Base plasticity rate (controlled by genetics/development)
        - oja_alpha: Weight decay strength (prevents saturation) - CRITICAL for stability
        - bcm_time_constant: How fast the sliding threshold adapts
        - dopamine_gating: Whether to require dopamine for learning
        - sparsity_penalty: Encourages sparse representations
        - max_weight_norm: Maximum L2 norm per output neuron (prevents explosion)
    """
    learning_rate: float = 0.001
    oja_alpha: float = 0.1  # INCREASED from 0.01 - stronger normalization prevents explosion
    bcm_time_constant: float = 100.0  # Steps to adapt threshold
    bcm_threshold_init: float = 0.5
    dopamine_gating: bool = True
    min_dopamine_for_learning: float = 0.3
    sparsity_penalty: float = 0.0001
    weight_clip: float = 2.0  # REDUCED from 10.0 - tighter clipping prevents gibberish
    max_weight_norm: float = 5.0  # NEW: Maximum L2 norm per output neuron row
    
    # STDP parameters (if using timing-based learning)
    stdp_tau_plus: float = 20.0  # ms
    stdp_tau_minus: float = 20.0  # ms
    stdp_a_plus: float = 0.005
    stdp_a_minus: float = 0.005
    
    # Debug settings
    debug_zero_activations: bool = True  # Print warning when activations are zero


# ============================================
# BCM SLIDING THRESHOLD
# ============================================

class BCMThreshold(nn.Module):
    """
    Bienenstock-Cooper-Munro sliding threshold for metaplasticity.
    
    Biological Parallel:
        The BCM rule states that the sign of plasticity (LTP vs LTD)
        depends on whether the postsynaptic activity exceeds a threshold.
        Crucially, this threshold ADAPTS based on recent activity history.
        
        Low activity → threshold decreases → easier to get LTP
        High activity → threshold increases → harder to get LTP
        
    This prevents:
        - Runaway potentiation (all weights → infinity)
        - Complete depression (all weights → 0)
        
    Mathematical Model:
        θ(t+1) = θ(t) + (1/τ) * (x²(t) - θ(t))
        
        Where τ is the time constant (higher = slower adaptation).
    """
    
    def __init__(
        self,
        num_neurons: int,
        time_constant: float = 100.0,
        initial_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.time_constant = time_constant
        
        # Per-neuron threshold (adapts independently)
        self.register_buffer(
            'threshold',
            torch.full((num_neurons,), initial_threshold)
        )
        
        # Track activity history for analysis
        self.register_buffer('activity_history', torch.zeros(100, num_neurons))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
    
    def update(self, post_activation: torch.Tensor):
        """
        Update threshold based on observed activity.
        
        Args:
            post_activation: Post-synaptic activations [batch, neurons]
        """
        # Average over batch
        if post_activation.dim() > 1:
            activity = post_activation.mean(0)
        else:
            activity = post_activation
        
        # BCM threshold update
        # θ ← θ + (1/τ) * (x² - θ)
        alpha = 1.0 / self.time_constant
        target = activity ** 2
        self.threshold = (1 - alpha) * self.threshold + alpha * target.detach()
        
        # Record history
        ptr = self.history_ptr.item()
        self.activity_history[ptr] = activity.detach()
        self.history_ptr = (self.history_ptr + 1) % 100
    
    def get_modification_factor(self, post_activation: torch.Tensor) -> torch.Tensor:
        """
        Get the BCM modification factor.
        
        Returns:
            factor [batch, neurons]: Positive for LTP, negative for LTD
               x > θ → positive (potentiation)
               x < θ → negative (depression)
        """
        # Broadcast threshold for batch
        if post_activation.dim() > 1:
            threshold = self.threshold.unsqueeze(0)
        else:
            threshold = self.threshold
        
        # BCM factor: x * (x - θ)
        factor = post_activation * (post_activation - threshold)
        
        return factor
    
    def reset(self, initial_threshold: float = 0.5):
        """Reset thresholds to initial value."""
        self.threshold.fill_(initial_threshold)
        self.activity_history.zero_()
        self.history_ptr.zero_()


# ============================================
# STDP TRACE
# ============================================

class STDPTrace(nn.Module):
    """
    Eligibility traces for Spike-Timing Dependent Plasticity.
    
    Biological Parallel:
        When a neuron fires, it leaves a "trace" of activation that
        decays exponentially over ~20ms. If the postsynaptic neuron
        fires while the presynaptic trace is still active, LTP occurs.
        
    Implementation:
        We use continuous activations as a proxy for spike trains.
        The trace follows: τ(t+1) = τ(t) * exp(-dt/τ_time) + x(t)
    """
    
    def __init__(
        self,
        num_neurons: int,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        super().__init__()
        
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        
        # Pre-synaptic trace (for detecting "pre before post")
        self.register_buffer('pre_trace', torch.zeros(num_neurons))
        
        # Post-synaptic trace (for detecting "post before pre")
        self.register_buffer('post_trace', torch.zeros(num_neurons))
    
    def update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        dt: float = 1.0,
    ):
        """
        Update eligibility traces.
        
        Args:
            pre_activation: Pre-synaptic activity [neurons]
            post_activation: Post-synaptic activity [neurons]
            dt: Time step (in arbitrary units)
        """
        # Decay traces
        pre_decay = math.exp(-dt / self.tau_plus)
        post_decay = math.exp(-dt / self.tau_minus)
        
        self.pre_trace = pre_decay * self.pre_trace + pre_activation.detach().mean(0)
        self.post_trace = post_decay * self.post_trace + post_activation.detach().mean(0)
    
    def get_stdp_update(
        self,
        weight: torch.Tensor,
        a_plus: float = 0.005,
        a_minus: float = 0.005,
    ) -> torch.Tensor:
        """
        Compute STDP weight update.
        
        Returns:
            delta_w [out, in]: Weight update matrix
        """
        # LTP: pre trace correlates with current post
        # (pre fired first, post is firing now = causal)
        ltp = a_plus * torch.outer(self.post_trace, self.pre_trace)
        
        # LTD: post trace correlates with current pre
        # (post fired first, pre is firing now = anti-causal)
        ltd = a_minus * torch.outer(self.pre_trace, self.post_trace)
        
        return ltp - ltd
    
    def reset(self):
        """Reset traces to zero."""
        self.pre_trace.zero_()
        self.post_trace.zero_()


# ============================================
# MAIN HEBBIAN LEARNING MODULE
# ============================================

class HebbianLearnerV2(nn.Module):
    """
    Real-Time Hebbian Learning Module with BCM Metaplasticity.
    
    ==========================================================================
    HOW TO USE (IMMEDIATE LEARNING)
    ==========================================================================
    
    During forward pass:
    
        # In your layer forward:
        pre_activation = x
        post_activation = self.projection(x)
        
        # Apply Hebbian update IMMEDIATELY (before backprop)
        self.hebbian.apply_update(
            weight=self.projection.weight,
            pre=pre_activation,
            post=post_activation,
            dopamine=current_dopamine_level,
        )
        
    This creates TRUE local learning - no waiting for loss computation!
    
    ==========================================================================
    FEATURES
    ==========================================================================
    
    1. Oja's Rule: Prevents weight explosion
    2. BCM Threshold: Adaptive metaplasticity
    3. Dopamine Gating: Only learn from rewarding experiences
    4. Weight Clipping: Additional stability guarantee
    5. Sparsity Penalty: Encourages efficient representations
    
    ==========================================================================
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[HebbianConfig] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or HebbianConfig()
        
        # BCM threshold for metaplasticity
        self.bcm = BCMThreshold(
            num_neurons=out_features,
            time_constant=self.config.bcm_time_constant,
            initial_threshold=self.config.bcm_threshold_init,
        )
        
        # STDP traces for timing-based learning
        self.stdp = STDPTrace(
            num_neurons=max(in_features, out_features),
            tau_plus=self.config.stdp_tau_plus,
            tau_minus=self.config.stdp_tau_minus,
        )
        
        # Statistics tracking
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_update_magnitude', torch.tensor(0.0))
        self.register_buffer('avg_dopamine', torch.tensor(0.5))
    
    def compute_update(
        self,
        weight: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        dopamine: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute Hebbian weight update.
        
        Implements: Δw = η · DA · BCM_factor · Oja_term
        
        Args:
            weight: Current weight matrix [out, in]
            pre: Pre-synaptic activations [batch, in]
            post: Post-synaptic activations [batch, out]
            dopamine: Current dopamine level (0 to 1)
            
        Returns:
            delta: Weight update [out, in]
        """
        cfg = self.config
        batch_size = pre.shape[0] if pre.dim() > 1 else 1
        
        # Ensure 2D
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        
        # Average over batch
        pre_mean = pre.mean(0)  # [in]
        post_mean = post.mean(0)  # [out]
        
        # =====================================
        # 1. BCM MODIFICATION FACTOR
        # =====================================
        
        # Update threshold based on activity
        self.bcm.update(post)
        
        # Get BCM factor (positive for LTP, negative for LTD)
        bcm_factor = self.bcm.get_modification_factor(post).mean(0)  # [out]
        
        # =====================================
        # 2. BASIC HEBBIAN TERM
        # =====================================
        
        # Outer product: correlation between pre and post
        # [out, in] = [out, 1] @ [1, in]
        hebbian = torch.outer(post_mean, pre_mean)
        
        # =====================================
        # 3. OJA'S NORMALIZATION TERM
        # =====================================
        
        # Prevents unbounded growth: w ← w - α * post² * w
        # [out, in] = [out, 1] * [out, in]
        post_squared = (post_mean ** 2).unsqueeze(1)  # [out, 1]
        oja_decay = cfg.oja_alpha * post_squared * weight
        
        # =====================================
        # 4. COMBINE WITH BCM
        # =====================================
        
        # Apply BCM factor (per output neuron)
        bcm_modulated = hebbian * bcm_factor.unsqueeze(1)  # [out, in]
        
        # =====================================
        # 5. DOPAMINE GATING
        # =====================================
        
        if cfg.dopamine_gating:
            # Dopamine gates learning
            # Low dopamine → no learning (protection from spurious updates)
            # High dopamine → enhanced learning
            da_gate = max(0, dopamine - cfg.min_dopamine_for_learning)
            da_gate = da_gate / (1 - cfg.min_dopamine_for_learning)  # Rescale to [0, 1]
            da_mult = da_gate * (0.5 + dopamine)  # Extra boost for high DA
        else:
            da_mult = 1.0
        
        # =====================================
        # 6. SPARSITY PENALTY
        # =====================================
        
        # Encourage sparse activations
        sparsity_term = cfg.sparsity_penalty * torch.sign(weight)
        
        # =====================================
        # 7. FINAL UPDATE
        # =====================================
        
        delta = cfg.learning_rate * da_mult * (bcm_modulated - oja_decay) - sparsity_term
        
        # Clip for stability
        delta = delta.clamp(-cfg.weight_clip, cfg.weight_clip)
        
        return delta
    
    def apply_update(
        self,
        weight: Union[torch.Tensor, nn.Parameter],
        pre: torch.Tensor,
        post: torch.Tensor,
        dopamine: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Apply Hebbian update to weight matrix IN-PLACE.
        
        This is the main method for IMMEDIATE learning.
        Call this during the forward pass to apply local learning
        BEFORE backpropagation runs.
        
        CRITICAL: Uses weight.data.add_() to work under torch.no_grad()!
        
        Args:
            weight: Weight matrix to update [out, in]
            pre: Pre-synaptic activations [batch, in]
            post: Post-synaptic activations [batch, out]
            dopamine: Current dopamine level
            mask: Optional sparsity mask [out, in]
            
        Returns:
            bool: True if update was applied, False if skipped (zero activations)
        """
        cfg = self.config
        
        # =====================================
        # DEBUG: Check for zero activations
        # =====================================
        if hasattr(cfg, 'debug_zero_activations') and cfg.debug_zero_activations:
            pre_sum = pre.abs().sum().item() if isinstance(pre, torch.Tensor) else 0
            post_sum = post.abs().sum().item() if isinstance(post, torch.Tensor) else 0
            
            if pre_sum < 1e-8:
                print(f"    ⚠️  [Hebbian] pre_activations are ZERO - Thalamus may be blocking signal!")
                return False
            if post_sum < 1e-8:
                print(f"    ⚠️  [Hebbian] post_activations are ZERO - No neural response!")
                return False
        
        with torch.no_grad():
            delta = self.compute_update(weight, pre, post, dopamine)
            
            # Apply mask if provided (for sparse layers)
            if mask is not None:
                delta = delta * mask
            
            # =====================================
            # FORCE IN-PLACE UPDATE (critical for no_grad mode)
            # =====================================
            if isinstance(weight, nn.Parameter):
                weight.data.add_(delta)
            else:
                weight.data.add_(delta) if hasattr(weight, 'data') else weight.add_(delta)
            
            # =====================================
            # POST-UPDATE NORMALIZATION (Oja's stability guarantee)
            # =====================================
            # Normalize each output neuron's weight vector to prevent explosion
            if hasattr(cfg, 'max_weight_norm'):
                weight_data = weight.data if isinstance(weight, nn.Parameter) else weight
                row_norms = weight_data.norm(dim=1, keepdim=True)
                scale = torch.clamp(cfg.max_weight_norm / (row_norms + 1e-8), max=1.0)
                weight_data.mul_(scale)
            
            # Track statistics
            self.update_count += 1
            # CRITICAL FIX: Extract scalar value to avoid CUDA/CPU device mismatch
            self.total_update_magnitude += delta.abs().mean().item()
            
            alpha = 0.01
            self.avg_dopamine = (1 - alpha) * self.avg_dopamine + alpha * dopamine
            
            return True
    
    def apply_clamped_update(
        self,
        weight: Union[torch.Tensor, nn.Parameter],
        pre: torch.Tensor,
        target_activation: torch.Tensor,
        dopamine: float = 1.0,
        learning_rate_override: Optional[float] = None,
        soft_clamp_strength: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Apply CLAMPED Hebbian update - Teacher Forcing for Synapses.
        
        =========================================================================
        V3: STABILIZED CLAMPED HEBBIAN WITH NORMALIZATION
        =========================================================================
        
        Key improvements over V2:
        1. L2 normalization of pre/target → constant energy signal
        2. Strict per-synapse delta clipping (max 0.05 per step)
        3. Lateral inhibition: decay competing weights slightly
        4. Soft forgetting for stability
        
        Biological Parallel:
            - Like a teacher guiding a student's hand to write the correct letter
            - The student's neurons are "clamped" to the correct activation
            - "What fires together, wires together" - but we FORCE the correct firing
            - Lateral inhibition prevents runaway excitation
            
        Args:
            weight: Weight matrix to update [out, in]
            pre: Pre-synaptic activations [batch, in]
            target_activation: TARGET post-synaptic activation [batch, out]
            dopamine: Current dopamine level (scales learning)
            learning_rate_override: Override the config learning rate
            soft_clamp_strength: How strongly to clamp (0=pure self, 1=pure target)
            
        Returns:
            Tuple[bool, float]: (success, total_weight_change)
        =========================================================================
        """
        cfg = self.config
        lr = learning_rate_override if learning_rate_override else cfg.learning_rate
        
        # =====================================
        # HYPERPARAMETERS FOR STABILITY
        # =====================================
        MAX_DELTA_PER_SYNAPSE = 0.05  # CRITICAL: prevents weight explosion
        LATERAL_INHIBITION_RATE = 0.001  # Soft decay for non-target weights
        PRE_NORM_SCALE = 1.0  # Target L2 norm for pre-activations
        TARGET_NORM_SCALE = 1.0  # Target L2 norm for target activations
        
        # Ensure tensors are on the same device as weight
        if isinstance(pre, torch.Tensor) and pre.device != weight.device:
            pre = pre.to(weight.device)
        if isinstance(target_activation, torch.Tensor) and target_activation.device != weight.device:
            target_activation = target_activation.to(weight.device)
        
        # Validate inputs
        pre_sum = pre.abs().sum().item() if isinstance(pre, torch.Tensor) else 0
        target_sum = target_activation.abs().sum().item() if isinstance(target_activation, torch.Tensor) else 0
        
        if pre_sum < 1e-8:
            return False, 0.0
        if target_sum < 1e-8:
            return False, 0.0
        
        with torch.no_grad():
            # Ensure 2D
            if pre.dim() == 1:
                pre = pre.unsqueeze(0)
            if target_activation.dim() == 1:
                target_activation = target_activation.unsqueeze(0)
            
            # Average over batch
            pre_mean = pre.mean(0)  # [in]
            target_mean = target_activation.mean(0)  # [out]
            
            # Ensure dimensions match weight
            out_features, in_features = weight.shape
            
            if pre_mean.numel() != in_features:
                if pre_mean.numel() > in_features:
                    pre_mean = pre_mean[:in_features]
                else:
                    pre_mean = F.pad(pre_mean, (0, in_features - pre_mean.numel()))
            
            if target_mean.numel() != out_features:
                if target_mean.numel() > out_features:
                    target_mean = target_mean[:out_features]
                else:
                    target_mean = F.pad(target_mean, (0, out_features - target_mean.numel()))
            
            # =====================================
            # 1. L2 NORMALIZATION OF INPUTS (CRITICAL)
            # =====================================
            # This ensures constant energy regardless of vector magnitude
            pre_norm = pre_mean.norm(p=2) + 1e-8
            target_norm = target_mean.norm(p=2) + 1e-8
            
            pre_normalized = (pre_mean / pre_norm) * PRE_NORM_SCALE
            target_normalized = (target_mean / target_norm) * TARGET_NORM_SCALE
            
            # =====================================
            # 2. CLAMPED HEBBIAN TERM (on normalized vectors)
            # =====================================
            # outer product now has bounded magnitude
            hebbian_term = torch.outer(target_normalized, pre_normalized)
            
            # =====================================
            # 3. OJA'S NORMALIZATION (additional stability)
            # =====================================
            # Decay proportional to activation squared
            oja_decay = cfg.oja_alpha * (target_normalized ** 2).unsqueeze(1) * weight.data.float()
            
            # =====================================
            # 4. DOPAMINE GATING
            # =====================================
            if cfg.dopamine_gating:
                da_gate = max(0, dopamine - cfg.min_dopamine_for_learning)
                da_gate = da_gate / (1 - cfg.min_dopamine_for_learning + 1e-8)
                # Reduce dopamine boost to prevent explosion
                da_mult = da_gate * (0.3 + 0.7 * dopamine)  # Max ~1.0 instead of ~1.4
            else:
                da_mult = 1.0
            
            # =====================================
            # 5. LATERAL INHIBITION (Soft Forgetting)
            # =====================================
            # Create a mask for target neurons (where target > threshold)
            target_mask = (target_normalized.abs() > 0.1).float().unsqueeze(1)  # [out, 1]
            
            # For non-target neurons, apply small decay (lateral inhibition)
            # This prevents other weights from staying strong when we want "blue"
            non_target_mask = 1.0 - target_mask
            lateral_inhibition = LATERAL_INHIBITION_RATE * non_target_mask * weight.data.float()
            
            # =====================================
            # 6. COMPUTE RAW DELTA
            # =====================================
            raw_delta = lr * da_mult * (hebbian_term - oja_decay) - lateral_inhibition
            
            # =====================================
            # 7. STRICT DELTA CLIPPING (CRITICAL FOR STABILITY)
            # =====================================
            # This is the key fix: limit change per synapse per step
            delta = torch.clamp(raw_delta, -MAX_DELTA_PER_SYNAPSE, MAX_DELTA_PER_SYNAPSE)
            
            # =====================================
            # 8. APPLY IN-PLACE UPDATE
            # =====================================
            if isinstance(weight, nn.Parameter):
                weight.data.add_(delta.to(weight.dtype))
            else:
                weight.data.add_(delta.to(weight.dtype))
            
            # =====================================
            # 9. POST-UPDATE WEIGHT CLIPPING (Hard bounds)
            # =====================================
            # Ensure weights stay in reasonable range
            weight.data.clamp_(-cfg.weight_clip, cfg.weight_clip)
            
            # =====================================
            # 10. POST-UPDATE NORMALIZATION (Row norm limit)
            # =====================================
            if hasattr(cfg, 'max_weight_norm'):
                weight_data = weight.data if isinstance(weight, nn.Parameter) else weight
                row_norms = weight_data.norm(dim=1, keepdim=True)
                scale = torch.clamp(cfg.max_weight_norm / (row_norms + 1e-8), max=1.0)
                weight_data.mul_(scale)
            
            # Track statistics
            total_change = delta.abs().sum().item()
            self.update_count += 1
            self.total_update_magnitude += delta.abs().mean().item()
            
            alpha = 0.01
            self.avg_dopamine = (1 - alpha) * self.avg_dopamine + alpha * dopamine
            
            return True, total_change

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        count = max(1, self.update_count.item())
        return {
            'update_count': self.update_count.item(),
            'avg_update_magnitude': (self.total_update_magnitude / count).item(),
            'avg_dopamine': self.avg_dopamine.item(),
            'bcm_threshold_mean': self.bcm.threshold.mean().item(),
            'bcm_threshold_std': self.bcm.threshold.std().item(),
        }
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.update_count.zero_()
        self.total_update_magnitude.zero_()
        self.bcm.reset()
        self.stdp.reset()


# ============================================
# HEBBIAN LINEAR LAYER
# ============================================

class HebbianLinear(nn.Module):
    """
    Linear layer with built-in immediate Hebbian learning.
    
    This is a drop-in replacement for nn.Linear that performs
    local learning during the forward pass.
    
    Usage:
        layer = HebbianLinear(256, 512, hebbian_lr=0.001)
        
        # In forward:
        output = layer(x, dopamine=current_da_level)
        
        # Learning happens automatically during forward!
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hebbian_lr: float = 0.001,
        dopamine_gating: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Hebbian learner
        self.hebbian = HebbianLearnerV2(
            in_features,
            out_features,
            HebbianConfig(
                learning_rate=hebbian_lr,
                dopamine_gating=dopamine_gating,
            ),
        )
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        
        # Store activations for potential use
        self.last_pre = None
        self.last_post = None
    
    def forward(
        self,
        x: torch.Tensor,
        dopamine: float = 1.0,
        learn: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional immediate Hebbian learning.
        
        Args:
            x: Input tensor [batch, in_features]
            dopamine: Current dopamine level for gating
            learn: Whether to apply Hebbian update
            
        Returns:
            output: Transformed tensor [batch, out_features]
        """
        # Standard linear transform
        output = F.linear(x, self.weight, self.bias)
        
        # Apply Hebbian learning during training
        if learn and self.training:
            self.hebbian.apply_update(
                weight=self.weight,
                pre=x,
                post=output,
                dopamine=dopamine,
            )
        
        # Store for potential external use
        self.last_pre = x.detach()
        self.last_post = output.detach()
        
        return output
    
    def get_hebbian_stats(self) -> Dict:
        """Get Hebbian learning statistics."""
        return self.hebbian.get_statistics()


# ============================================
# CORTEX INTEGRATION HELPER
# ============================================

class CorticalHebbianIntegrator(nn.Module):
    """
    Helper class to integrate Hebbian learning into existing cortical layers.
    
    This wraps an existing linear layer and adds Hebbian learning without
    modifying the original architecture.
    
    Usage:
        # Wrap existing layer
        integrator = CorticalHebbianIntegrator(
            existing_layer=my_linear_layer,
            hebbian_lr=0.001,
        )
        
        # During forward:
        output = integrator.forward_with_hebbian(x, dopamine)
    """
    
    def __init__(
        self,
        existing_layer: nn.Linear,
        hebbian_lr: float = 0.001,
        dopamine_gating: bool = True,
    ):
        super().__init__()
        
        self.layer = existing_layer
        
        self.hebbian = HebbianLearnerV2(
            existing_layer.in_features,
            existing_layer.out_features,
            HebbianConfig(
                learning_rate=hebbian_lr,
                dopamine_gating=dopamine_gating,
            ),
        )
    
    def forward_with_hebbian(
        self,
        x: torch.Tensor,
        dopamine: float = 1.0,
    ) -> torch.Tensor:
        """Forward with immediate Hebbian learning."""
        output = self.layer(x)
        
        if self.training:
            self.hebbian.apply_update(
                weight=self.layer.weight,
                pre=x,
                post=output,
                dopamine=dopamine,
            )
        
        return output


# ============================================
# BACKWARDS COMPATIBILITY
# ============================================

# Alias for existing code
HebbianPlasticity = HebbianLearnerV2
