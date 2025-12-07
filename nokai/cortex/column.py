"""
Cortical Column - The fundamental processing unit of Nōkai

Inspired by the minicolumns of the neocortex, each column is a 
specialized processing unit that learns to represent specific features.
Columns communicate laterally with sparse connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from nokai.config import CorticalColumnConfig


class SparseLinear(nn.Module):
    """
    Linear layer with structured sparsity for efficiency.
    Only a fraction of connections are active.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.9,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Full weight matrix (will be masked)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Create sparse mask (non-learnable)
        mask = torch.rand(out_features, in_features) > sparsity
        self.register_buffer('mask', mask.float())
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # Scale by sparsity to maintain output variance
        with torch.no_grad():
            self.weight.mul_(1.0 / (1.0 - self.sparsity + 1e-8))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply sparse mask
        sparse_weight = self.weight * self.mask
        return F.linear(x, sparse_weight, self.bias)
    
    def update_mask(self, importance_scores: Optional[torch.Tensor] = None):
        """
        Dynamically update sparsity mask based on importance.
        Implements magnitude-based pruning with regrowth.
        """
        with torch.no_grad():
            if importance_scores is None:
                importance_scores = self.weight.abs()
            
            # Flatten scores
            flat_scores = importance_scores.flatten()
            k = int((1 - self.sparsity) * flat_scores.numel())
            
            # Top-k selection
            _, indices = torch.topk(flat_scores, k)
            new_mask = torch.zeros_like(flat_scores)
            new_mask[indices] = 1.0
            
            self.mask.copy_(new_mask.view_as(self.mask))


class NeuronPopulation(nn.Module):
    """
    A population of neurons within a cortical column.
    Implements biologically-inspired activation dynamics.
    """
    
    def __init__(
        self,
        num_neurons: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        
        # Neuron parameters
        self.membrane_potential = nn.Parameter(torch.zeros(num_neurons), requires_grad=False)
        self.threshold = nn.Parameter(torch.ones(num_neurons), requires_grad=False)
        
        # Activation function
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(num_neurons)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),  # Swish
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),  # More biologically plausible
        }
        return activations.get(name, nn.GELU())
    
    def forward(
        self,
        x: torch.Tensor,
        modulation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through neuron population.
        
        Args:
            x: Input tensor [batch, neurons]
            modulation: Optional modulatory signal (e.g., from oscillations)
            
        Returns:
            output: Activated output
            sparsity_mask: Binary mask of active neurons
        """
        # Apply modulation if provided
        if modulation is not None:
            x = x * modulation
        
        # Normalize
        x = self.norm(x)
        
        # Activation
        x = self.activation(x)
        
        # Sparse activation - only top-k neurons fire
        # This mimics biological sparse coding
        k = max(1, int(0.1 * self.num_neurons))  # 10% sparsity
        values, indices = torch.topk(x, k, dim=-1)
        
        sparsity_mask = torch.zeros_like(x)
        sparsity_mask.scatter_(-1, indices, 1.0)
        
        # Apply sparse mask
        x = x * sparsity_mask
        
        # Dropout for regularization
        x = self.dropout(x)
        
        return x, sparsity_mask


class CorticalColumn(nn.Module):
    """
    A single cortical column - the fundamental processing unit.
    
    Implements:
        - Feedforward processing (bottom-up)
        - Feedback predictions (top-down)
        - Lateral connections (horizontal)
        - Sparse activation
    """
    
    def __init__(
        self,
        config: CorticalColumnConfig,
        column_id: int = 0,
        input_dim: int = 256,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.column_id = column_id
        self.input_dim = input_dim
        self.output_dim = output_dim or config.num_neurons
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, config.num_neurons)
        
        # Neuron populations for each layer
        self.layers = nn.ModuleList([
            NeuronPopulation(
                config.num_neurons,
                config.activation,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Feedforward connections between layers
        self.feedforward = nn.ModuleList([
            SparseLinear(
                config.num_neurons,
                config.num_neurons,
                sparsity=1.0 - config.lateral_connectivity,
            )
            for _ in range(config.num_layers - 1)
        ])
        
        # Top-down (feedback) connections for predictive coding
        self.feedback = nn.ModuleList([
            SparseLinear(
                config.num_neurons,
                config.num_neurons,
                sparsity=0.95,  # Very sparse feedback
            )
            for _ in range(config.num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.num_neurons, self.output_dim)
        
        # Lateral connection placeholders (connected during cortex assembly)
        self.lateral_inputs = nn.ModuleDict()
        
        # Column state for temporal processing
        self.register_buffer('state', torch.zeros(1, config.num_neurons))
        
        # Tracking for Hebbian learning
        self.register_buffer('pre_activations', torch.zeros(config.num_layers, config.num_neurons))
        self.register_buffer('post_activations', torch.zeros(config.num_layers, config.num_neurons))
    
    def add_lateral_connection(self, source_id: int, weight: nn.Linear):
        """Add a lateral connection from another column."""
        self.lateral_inputs[str(source_id)] = weight
    
    def forward(
        self,
        x: torch.Tensor,
        lateral_inputs: Optional[dict] = None,
        top_down_prediction: Optional[torch.Tensor] = None,
        oscillation_phase: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process input through the cortical column.
        
        Args:
            x: Input tensor [batch, input_dim]
            lateral_inputs: Dict of inputs from connected columns
            top_down_prediction: Prediction from higher-level column
            oscillation_phase: Current phase of neural oscillation
            
        Returns:
            output: Column output
            metadata: Dict with activations, errors, etc.
        """
        batch_size = x.shape[0]
        
        # Project input
        h = self.input_proj(x)
        
        # Add lateral inputs if available
        if lateral_inputs:
            for source_id, lateral_x in lateral_inputs.items():
                if str(source_id) in self.lateral_inputs:
                    h = h + self.lateral_inputs[str(source_id)](lateral_x)
        
        # Calculate oscillation modulation if phase provided
        modulation = None
        if oscillation_phase is not None:
            # Gamma-band modulation (fast oscillation within theta cycle)
            modulation = torch.cos(torch.tensor(oscillation_phase * 2 * 3.14159))
            modulation = 0.5 + 0.5 * modulation  # Map to [0, 1]
        
        # Process through layers
        layer_outputs = []
        sparsity_masks = []
        prediction_errors = []
        
        for i, (layer, ff) in enumerate(zip(self.layers[:-1], self.feedforward)):
            # Store pre-activation for Hebbian learning
            self.pre_activations[i] = h.detach().mean(0)
            
            # Apply layer
            h, mask = layer(h, modulation)
            layer_outputs.append(h)
            sparsity_masks.append(mask)
            
            # Store post-activation
            self.post_activations[i] = h.detach().mean(0)
            
            # Feedforward to next layer
            h = ff(h)
            
            # Compute prediction error if top-down available
            if top_down_prediction is not None and i < len(self.feedback):
                prediction = self.feedback[i](layer_outputs[-1] if layer_outputs else h)
                error = h - prediction
                prediction_errors.append(error)
        
        # Final layer
        h, mask = self.layers[-1](h, modulation)
        layer_outputs.append(h)
        sparsity_masks.append(mask)
        self.pre_activations[-1] = h.detach().mean(0)
        self.post_activations[-1] = h.detach().mean(0)
        
        # Output projection
        output = self.output_proj(h)
        
        # Update state for temporal continuity
        self.state = h.detach().mean(0, keepdim=True)
        
        # Compile metadata
        metadata = {
            'layer_outputs': layer_outputs,
            'sparsity_masks': sparsity_masks,
            'prediction_errors': prediction_errors,
            'mean_sparsity': sum(m.mean().item() for m in sparsity_masks) / len(sparsity_masks),
            'column_id': self.column_id,
        }
        
        return output, metadata
    
    def get_hebbian_update(self, learning_rate: float = 0.001) -> dict:
        """
        Compute Hebbian weight updates based on pre/post activations.
        
        Implements simplified STDP (Spike-Timing Dependent Plasticity):
            Δw = η * pre * post - λ * w  (with normalization)
        """
        updates = {}
        
        for i, ff in enumerate(self.feedforward):
            pre = self.pre_activations[i]
            post = self.post_activations[i + 1]
            
            # Outer product gives correlation
            delta = learning_rate * torch.outer(post, pre)
            
            # Weight decay term
            delta = delta - 0.0001 * ff.weight.data
            
            # Apply sparsity mask
            delta = delta * ff.mask
            
            updates[f'feedforward_{i}'] = delta
        
        return updates
    
    def apply_hebbian_update(self, updates: dict):
        """Apply pre-computed Hebbian updates to weights."""
        with torch.no_grad():
            for name, delta in updates.items():
                if name.startswith('feedforward_'):
                    idx = int(name.split('_')[1])
                    self.feedforward[idx].weight.data += delta
