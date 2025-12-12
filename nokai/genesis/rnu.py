"""
Rich Neuron Unit (RNU) - Dynamic Neurons with Internal State

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

Real neurons are NOT simple σ(Wx + b) units!

They have:
1. MEMBRANE POTENTIAL: Integrates inputs over time (leaky integration)
2. ADAPTIVE THRESHOLD: Harder to fire after recent activity
3. REFRACTORY PERIOD: Cannot fire immediately after a spike
4. ELIGIBILITY TRACES: Memory of recent pre/post activity for learning
5. STOCHASTIC FIRING: Probabilistic output based on membrane state

This module implements all of these dynamics efficiently on GPU.

=============================================================================
MATHEMATICAL MODEL
=============================================================================

Membrane dynamics (Leaky Integrate-and-Fire):
    v(t+1) = τ_m · v(t) + (1-τ_m) · [I(t) - θ_adapt(t)]

Threshold adaptation:
    θ_adapt(t+1) = τ_θ · θ_adapt(t) + (1-τ_θ) · [θ_base + β·s(t)]

Refractory period:
    f(t+1) = max(0, f(t) - δ_f + α_f · |s(t)|)

Spike probability:
    p(t) = sigmoid(v(t) - θ_adapt(t)) · (1 - f(t))

Stochastic output (ternary):
    s(t) ~ Bernoulli(p) * Sign(v(t))

Eligibility trace (for STDP):
    e(t+1) = λ·e(t) + s(t) ⊗ x_pre(t)

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class RNUConfig:
    """Configuration for Rich Neuron Units."""
    
    # Membrane dynamics
    tau_membrane: float = 0.9       # Membrane time constant
    tau_threshold: float = 0.99    # Threshold adaptation speed
    theta_base: float = 1.0         # Base firing threshold
    beta: float = 0.1               # Threshold adaptation strength
    
    # Refractory period
    delta_fatigue: float = 0.1      # Fatigue recovery rate
    alpha_fatigue: float = 0.5      # Fatigue increase per spike
    
    # Eligibility trace
    lambda_trace: float = 0.95      # Trace decay rate
    
    # Output
    stochastic: bool = True         # Enable stochastic firing
    ternary_output: bool = True     # Output ∈ {-1, 0, +1}
    
    # Regularization
    target_sparsity: float = 0.95   # Target fraction of silent neurons


class RichNeuronUnit(nn.Module):
    """
    A population of Rich Neurons with dynamic internal state.
    
    ═══════════════════════════════════════════════════════════════════════
    KEY DIFFERENCES FROM STANDARD NEURONS
    ═══════════════════════════════════════════════════════════════════════
    
    Standard: y = σ(Wx + b)
        - Stateless (no memory between forwards)
        - Deterministic
        - Fixed threshold (implicit in activation)
        
    RNU: y ~ Bernoulli(σ(v - θ)) * sign(v)
        - Maintains state (membrane, threshold, fatigue, trace)
        - Stochastic output
        - Adaptive threshold
        - Eligibility traces for local learning
    
    ═══════════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        num_neurons: int,
        input_dim: int,
        config: Optional[RNUConfig] = None,
        use_ternary_weights: bool = True,
    ):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.config = config or RNUConfig()
        self.use_ternary_weights = use_ternary_weights
        
        # ═══════════════════════════════════════════════════════════════
        # SYNAPTIC WEIGHTS
        # ═══════════════════════════════════════════════════════════════
        if use_ternary_weights:
            from nokai.genesis.ternary import TernaryLinear
            self.synapses = TernaryLinear(input_dim, num_neurons, bias=False)
        else:
            self.synapses = nn.Linear(input_dim, num_neurons, bias=False)
        
        # ═══════════════════════════════════════════════════════════════
        # INTERNAL STATE (persistent across forward calls)
        # ═══════════════════════════════════════════════════════════════
        
        # Membrane potential
        self.register_buffer('membrane', torch.zeros(num_neurons))
        
        # Adaptive threshold
        self.register_buffer('threshold', torch.full((num_neurons,), config.theta_base if config else 1.0))
        
        # Refractory/fatigue
        self.register_buffer('fatigue', torch.zeros(num_neurons))
        
        # Eligibility trace for STDP
        self.register_buffer('eligibility_trace', torch.zeros(num_neurons, input_dim))
        
        # Last input (for trace computation)
        self.register_buffer('last_input', torch.zeros(input_dim))
        
        # Last output (for learning)
        self.register_buffer('last_output', torch.zeros(num_neurons))
        
        # ═══════════════════════════════════════════════════════════════
        # STATISTICS
        # ═══════════════════════════════════════════════════════════════
        self.register_buffer('firing_rate', torch.zeros(num_neurons))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
    
    def reset_state(self):
        """Reset all internal state to initial values."""
        cfg = self.config
        
        self.membrane.zero_()
        self.threshold.fill_(cfg.theta_base)
        self.fatigue.zero_()
        self.eligibility_trace.zero_()
        self.last_input.zero_()
        self.last_output.zero_()
        self.firing_rate.zero_()
        self.step_count.zero_()
    
    def forward(
        self,
        x: torch.Tensor,
        neuromodulation: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through RNU population.
        
        Args:
            x: Input tensor [batch, input_dim] or [input_dim]
            neuromodulation: Optional dict with 'acetylcholine', 'dopamine', etc.
            return_state: If True, return internal state dict
            
        Returns:
            output: Spike tensor [batch, num_neurons] ∈ {-1, 0, +1}
            state: Optional dict of internal state
        """
        cfg = self.config
        
        # Handle batched or single input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Expand state for batch
        membrane = self.membrane.unsqueeze(0).expand(batch_size, -1)
        threshold = self.threshold.unsqueeze(0).expand(batch_size, -1)
        fatigue = self.fatigue.unsqueeze(0).expand(batch_size, -1)
        
        # ═══════════════════════════════════════════════════════════════
        # 1. SYNAPTIC INPUT
        # ═══════════════════════════════════════════════════════════════
        synaptic_input = self.synapses(x)  # [batch, num_neurons]
        
        # ═══════════════════════════════════════════════════════════════
        # 2. MEMBRANE DYNAMICS
        # ═══════════════════════════════════════════════════════════════
        # v(t+1) = τ·v(t) + (1-τ)·[I(t) - θ(t)]
        membrane = (
            cfg.tau_membrane * membrane + 
            (1 - cfg.tau_membrane) * (synaptic_input - threshold)
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 3. SPIKE PROBABILITY
        # ═══════════════════════════════════════════════════════════════
        # p = σ(v - θ) * (1 - fatigue)
        spike_prob = torch.sigmoid(membrane) * (1 - fatigue)
        
        # Neuromodulation affects spike probability
        if neuromodulation is not None:
            if 'acetylcholine' in neuromodulation:
                # ACh enhances responsiveness
                spike_prob = spike_prob * (0.5 + 0.5 * neuromodulation['acetylcholine'])
            if 'norepinephrine' in neuromodulation:
                # NE lowers threshold (increases probability)
                spike_prob = spike_prob + 0.1 * neuromodulation['norepinephrine']
        
        spike_prob = spike_prob.clamp(0, 1)
        
        # ═══════════════════════════════════════════════════════════════
        # 4. STOCHASTIC SPIKE GENERATION
        # ═══════════════════════════════════════════════════════════════
        if cfg.stochastic and self.training:
            # Sample spikes stochastically
            spike_mask = (torch.rand_like(spike_prob) < spike_prob).float()
        else:
            # Deterministic (threshold at 0.5)
            spike_mask = (spike_prob > 0.5).float()
        
        # ═══════════════════════════════════════════════════════════════
        # 5. TERNARY OUTPUT
        # ═══════════════════════════════════════════════════════════════
        if cfg.ternary_output:
            # Sign from membrane potential
            output = spike_mask * torch.sign(membrane)
        else:
            # Binary (just magnitude)
            output = spike_mask
        
        # ═══════════════════════════════════════════════════════════════
        # 6. UPDATE INTERNAL STATE
        # ═══════════════════════════════════════════════════════════════
        
        # Average over batch for state update
        mean_output = output.mean(0)
        mean_input = x.mean(0)
        
        # Threshold adaptation: increases after firing
        self.threshold = (
            cfg.tau_threshold * self.threshold +
            (1 - cfg.tau_threshold) * (cfg.theta_base + cfg.beta * mean_output.abs())
        )
        
        # Fatigue update
        self.fatigue = torch.clamp(
            self.fatigue - cfg.delta_fatigue + cfg.alpha_fatigue * mean_output.abs(),
            min=0.0, max=0.9
        )
        
        # Update membrane (store mean for next step)
        self.membrane = membrane.mean(0)
        
        # ═══════════════════════════════════════════════════════════════
        # 7. ELIGIBILITY TRACE (for STDP)
        # ═══════════════════════════════════════════════════════════════
        # e(t+1) = λ·e(t) + post ⊗ pre
        self.eligibility_trace = (
            cfg.lambda_trace * self.eligibility_trace +
            torch.outer(mean_output, mean_input)
        )
        
        # Store for external access
        self.last_input = mean_input.detach()
        self.last_output = mean_output.detach()
        
        # ═══════════════════════════════════════════════════════════════
        # 8. STATISTICS UPDATE
        # ═══════════════════════════════════════════════════════════════
        self.step_count += 1
        alpha = 0.01
        current_rate = mean_output.abs().clamp(0, 1)
        self.firing_rate = (1 - alpha) * self.firing_rate + alpha * current_rate
        
        # Build state dict if requested
        state = None
        if return_state:
            state = {
                'membrane': membrane.detach(),
                'threshold': threshold.detach(),
                'fatigue': fatigue.detach(),
                'spike_prob': spike_prob.detach(),
                'firing_rate': self.firing_rate.detach(),
                'mean_sparsity': 1 - mean_output.abs().mean().item(),
            }
        
        return output, state
    
    def get_eligibility_trace(self) -> torch.Tensor:
        """Get current eligibility trace for STDP learning."""
        return self.eligibility_trace
    
    def get_firing_rate(self) -> torch.Tensor:
        """Get estimated firing rate per neuron."""
        return self.firing_rate
    
    def get_sparsity(self) -> float:
        """Get current sparsity (fraction of silent neurons)."""
        return 1 - self.firing_rate.mean().item()


class RNULayer(nn.Module):
    """
    A layer of RNU neurons with lateral connections.
    
    Implements lateral inhibition (winner-take-all dynamics) 
    for sparse coding.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        config: Optional[RNUConfig] = None,
        lateral_inhibition: float = 0.1,
        use_ternary: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.lateral_inhibition = lateral_inhibition
        
        # Main RNU population
        self.rnu = RichNeuronUnit(
            num_neurons=num_neurons,
            input_dim=input_dim,
            config=config,
            use_ternary_weights=use_ternary,
        )
        
        # Lateral connections (inhibitory)
        if lateral_inhibition > 0:
            self.lateral = nn.Linear(num_neurons, num_neurons, bias=False)
            # Initialize as negative (inhibitory)
            nn.init.uniform_(self.lateral.weight, -0.1, 0)
            # Zero self-connections
            with torch.no_grad():
                self.lateral.weight.fill_diagonal_(0)
        else:
            self.lateral = None
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(num_neurons)
    
    def forward(
        self,
        x: torch.Tensor,
        neuromodulation: Optional[Dict[str, torch.Tensor]] = None,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """
        Forward with lateral inhibition dynamics.
        
        Args:
            x: Input [batch, input_dim]
            neuromodulation: Modulatory signals
            n_iterations: Number of lateral inhibition iterations
            
        Returns:
            output: Sparse output [batch, num_neurons]
        """
        # Initial forward
        output, _ = self.rnu(x, neuromodulation)
        
        # Lateral inhibition iterations (optional)
        if self.lateral is not None:
            for _ in range(n_iterations):
                # Inhibitory lateral input
                lateral_input = self.lateral(output.abs())
                
                # Reduce activity where inhibition is strong
                inhibition_mask = torch.sigmoid(-lateral_input * 2)
                output = output * inhibition_mask
        
        # Normalize
        output = self.norm(output)
        
        return output
    
    def reset_state(self):
        """Reset RNU state."""
        self.rnu.reset_state()


class RNUCorticalColumn(nn.Module):
    """
    A stack of RNU layers forming a cortical column.
    
    Implements feedforward + feedback (predictive coding) dynamics.
    """
    
    def __init__(
        self,
        input_dim: int,
        layer_dims: Tuple[int, ...] = (256, 256, 256),
        config: Optional[RNUConfig] = None,
        use_feedback: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.use_feedback = use_feedback
        
        # Build layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(layer_dims)
        
        for i in range(self.num_layers):
            self.layers.append(
                RNULayer(
                    input_dim=dims[i],
                    num_neurons=dims[i + 1],
                    config=config,
                )
            )
        
        # Feedback projections (top-down predictions)
        if use_feedback:
            self.feedback = nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.feedback.append(
                    nn.Linear(dims[i + 2], dims[i + 1], bias=False)
                )
        
        # Output projection
        self.output_dim = layer_dims[-1]
    
    def forward(
        self,
        x: torch.Tensor,
        neuromodulation: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward through cortical column.
        
        Returns:
            output: Top layer activation
            activations: Dict of all layer activations
        """
        activations = {}
        current = x
        
        # Bottom-up pass
        for i, layer in enumerate(self.layers):
            current = layer(current, neuromodulation)
            activations[f'layer_{i}'] = current
        
        # Top-down predictions (if enabled)
        predictions = {}
        if self.use_feedback and len(self.layers) > 1:
            for i in range(self.num_layers - 2, -1, -1):
                higher = activations[f'layer_{i + 1}']
                prediction = self.feedback[i](higher)
                predictions[f'prediction_{i}'] = prediction
                
                # Compute prediction error
                actual = activations[f'layer_{i}']
                error = actual - prediction
                activations[f'error_{i}'] = error
        
        return current, {**activations, **predictions}
    
    def reset_state(self):
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset_state()
    
    def get_all_eligibility_traces(self) -> Dict[str, torch.Tensor]:
        """Get eligibility traces from all layers."""
        traces = {}
        for i, layer in enumerate(self.layers):
            traces[f'layer_{i}'] = layer.rnu.get_eligibility_trace()
        return traces
