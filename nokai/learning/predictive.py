"""
Predictive Coding - Learn via prediction errors

The brain is a prediction machine. Learning happens by
minimizing prediction errors at each level of the hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PredictiveCodingLayer(nn.Module):
    """
    Predictive coding layer.
    
    Implements:
    - Top-down predictions
    - Bottom-up error signals
    - Learning via error minimization
    
    Based on Rao & Ballard (1999) predictive coding framework.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_iterations: int = 3,
        learning_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        
        # Generative model: hidden → prediction
        self.generator = nn.Linear(hidden_dim, input_dim)
        
        # Recognition model: input → hidden (for initialization)
        self.recognizer = nn.Linear(input_dim, hidden_dim)
        
        # Precision (inverse variance) - learned
        self.log_precision = nn.Parameter(torch.zeros(input_dim))
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        top_down: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Process input through predictive coding dynamics.
        
        Args:
            x: Bottom-up input [batch, input_dim]
            top_down: Optional top-down prediction
            
        Returns:
            hidden: Inferred hidden state
            prediction: Generated prediction
            metadata: Errors and other info
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state via recognition
        hidden = self.recognizer(x)
        hidden = self.norm(hidden)
        
        # Iterative inference
        errors = []
        for i in range(self.num_iterations):
            # Generate prediction
            prediction = self.generator(hidden)
            
            # Compute prediction error
            error = x - prediction
            errors.append(error.detach())
            
            # Weight by precision
            precision = F.softplus(self.log_precision)
            weighted_error = error * precision
            
            # Update hidden state to reduce error
            # Gradient of error w.r.t. hidden
            grad = torch.autograd.grad(
                (weighted_error ** 2).sum(),
                hidden,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            
            hidden = hidden - self.learning_rate * grad
            hidden = self.norm(hidden)
        
        # Final prediction
        prediction = self.generator(hidden)
        final_error = x - prediction
        
        metadata = {
            'prediction_error': final_error.abs().mean().item(),
            'error_reduction': (errors[0].abs().mean() - errors[-1].abs().mean()).item() 
                              if len(errors) > 1 else 0,
            'precision_mean': precision.mean().item(),
        }
        
        return hidden, prediction, metadata
    
    def learn_from_error(self, error: torch.Tensor, hidden: torch.Tensor):
        """
        Update weights based on prediction error.
        
        Uses a simplified form of predictive coding learning:
        Update generator to reduce prediction error.
        """
        with torch.no_grad():
            # Update generator weights
            # Δw = learning_rate * error ⊗ hidden
            delta = self.learning_rate * torch.einsum('bi,bh->ih', error, hidden) / error.shape[0]
            self.generator.weight.data.add_(delta)


class HierarchicalPredictiveCoding(nn.Module):
    """
    Stack of predictive coding layers forming a hierarchy.
    
    Higher levels predict lower levels, errors propagate upward.
    """
    
    def __init__(
        self,
        dims: list,
        num_iterations: int = 3,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(
                input_dim=dims[i],
                hidden_dim=dims[i + 1],
                num_iterations=num_iterations,
            )
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Process through hierarchy."""
        hidden_states = []
        all_metadata = []
        
        current = x
        for layer in self.layers:
            hidden, prediction, metadata = layer(current)
            hidden_states.append(hidden)
            all_metadata.append(metadata)
            current = hidden
        
        return hidden_states[-1], {
            'layer_metadata': all_metadata,
            'total_error': sum(m['prediction_error'] for m in all_metadata),
        }
