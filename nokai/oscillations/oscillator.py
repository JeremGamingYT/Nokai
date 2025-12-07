"""
Neural Oscillator - Brain wave synchronization

Implements the Kuramoto model for neural synchronization,
enabling coordination across processing units.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class NeuralOscillator(nn.Module):
    """
    Single neural oscillator with phase dynamics.
    
    Based on the Kuramoto model:
    dθ/dt = ω + Σ K_ij * sin(θ_j - θ_i)
    """
    
    def __init__(
        self,
        natural_frequency: float = 10.0,
        coupling_strength: float = 0.5,
    ):
        super().__init__()
        self.natural_frequency = natural_frequency
        self.coupling_strength = coupling_strength
        
        # Phase as buffer (not trained via gradient)
        self.register_buffer('phase', torch.tensor(0.0))
    
    def step(self, dt: float, external_phases: Optional[torch.Tensor] = None) -> float:
        """Advance oscillator by one time step."""
        # Natural frequency contribution
        dphase = 2 * math.pi * self.natural_frequency * dt
        
        # Coupling contribution
        if external_phases is not None:
            phase_diff = external_phases - self.phase
            coupling = self.coupling_strength * torch.sin(phase_diff).mean()
            dphase += coupling * dt
        
        self.phase = (self.phase + dphase) % (2 * math.pi)
        return self.phase.item()
    
    def get_modulation(self) -> torch.Tensor:
        """Get current modulation signal (0 to 1)."""
        return 0.5 + 0.5 * torch.cos(self.phase)


class OscillatorNetwork(nn.Module):
    """
    Network of coupled neural oscillators.
    
    Implements different frequency bands:
    - Theta (4-8 Hz): Sequential processing, memory encoding
    - Gamma (30-100 Hz): Feature binding, attention
    """
    
    def __init__(
        self,
        num_oscillators: int = 100,
        theta_freq: float = 6.0,
        gamma_freq: float = 40.0,
        coupling_strength: float = 0.3,
    ):
        super().__init__()
        self.num_oscillators = num_oscillators
        
        # Theta oscillators (slow, global coordination)
        self.theta_freq = theta_freq
        
        # Gamma oscillators (fast, local processing)
        self.gamma_freq = gamma_freq
        
        # Phases for all oscillators
        self.register_buffer('theta_phases', torch.rand(num_oscillators) * 2 * math.pi)
        self.register_buffer('gamma_phases', torch.rand(num_oscillators) * 2 * math.pi)
        
        # Coupling matrix (learnable sparse connections)
        self.coupling = nn.Parameter(torch.randn(num_oscillators, num_oscillators) * 0.1)
        
        # Mask for sparsity
        mask = torch.rand(num_oscillators, num_oscillators) > 0.9
        self.register_buffer('coupling_mask', mask.float())
        
        self.coupling_strength = coupling_strength
    
    def step(self, dt: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance all oscillators by one time step."""
        
        # Theta dynamics (slower)
        theta_coupling = torch.sin(
            self.theta_phases.unsqueeze(1) - self.theta_phases.unsqueeze(0)
        )
        theta_coupling = (theta_coupling * self.coupling * self.coupling_mask).sum(dim=1)
        
        d_theta = (
            2 * math.pi * self.theta_freq * dt +
            self.coupling_strength * theta_coupling * dt
        )
        self.theta_phases = (self.theta_phases + d_theta) % (2 * math.pi)
        
        # Gamma dynamics (faster, modulated by theta)
        theta_modulation = 0.5 + 0.5 * torch.cos(self.theta_phases)
        
        gamma_coupling = torch.sin(
            self.gamma_phases.unsqueeze(1) - self.gamma_phases.unsqueeze(0)
        )
        gamma_coupling = (gamma_coupling * self.coupling * self.coupling_mask).sum(dim=1)
        
        d_gamma = (
            2 * math.pi * self.gamma_freq * dt * theta_modulation +
            self.coupling_strength * gamma_coupling * dt
        )
        self.gamma_phases = (self.gamma_phases + d_gamma) % (2 * math.pi)
        
        return self.theta_phases, self.gamma_phases
    
    def get_modulation(self, oscillator_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get modulation signals for specified oscillators."""
        if oscillator_ids is None:
            theta_mod = 0.5 + 0.5 * torch.cos(self.theta_phases)
            gamma_mod = 0.5 + 0.5 * torch.cos(self.gamma_phases)
        else:
            theta_mod = 0.5 + 0.5 * torch.cos(self.theta_phases[oscillator_ids])
            gamma_mod = 0.5 + 0.5 * torch.cos(self.gamma_phases[oscillator_ids])
        
        # Combine: gamma nested in theta
        return theta_mod * gamma_mod
    
    def get_phase_coherence(self) -> float:
        """Measure synchronization level (0 = incoherent, 1 = synchronized)."""
        # Order parameter from Kuramoto model
        z_theta = torch.exp(1j * self.theta_phases.cdouble()).mean()
        z_gamma = torch.exp(1j * self.gamma_phases.cdouble()).mean()
        
        return (z_theta.abs().item() + z_gamma.abs().item()) / 2
    
    def reset(self):
        """Reset phases to random initial conditions."""
        self.theta_phases = torch.rand_like(self.theta_phases) * 2 * math.pi
        self.gamma_phases = torch.rand_like(self.gamma_phases) * 2 * math.pi
