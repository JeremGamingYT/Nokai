"""
Cortex Assembly - Complete cortical processing system
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from nokai.config import NokaiConfig
from nokai.cortex.layer import CorticalLayer


class Cortex(nn.Module):
    """Complete cortical processing system with hierarchical layers."""
    
    def __init__(self, config: NokaiConfig):
        super().__init__()
        self.config = config
        
        # Create hierarchical layers (pyramid structure)
        columns_per_layer = [
            config.num_columns,
            config.num_columns // 2,
            config.num_columns // 4,
        ]
        
        self.layers = nn.ModuleList([
            CorticalLayer(
                num_columns=n_cols,
                column_config=config.column_config,
                input_dim=config.embedding_dim,
                thalamus_config=config.thalamus,
            )
            for n_cols in columns_per_layer
        ])
        
        # Top-down prediction networks
        self.predictors = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.embedding_dim)
            for _ in range(len(columns_per_layer) - 1)
        ])
        
        self.output_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        oscillation_phase: Optional[float] = None,
        num_iterations: int = 1,
    ) -> Tuple[torch.Tensor, Dict]:
        """Process through hierarchical cortex with iterative refinement."""
        
        # Bottom-up pass
        layer_outputs = [x]
        layer_metadata = []
        
        for layer in self.layers:
            h, meta = layer(layer_outputs[-1], oscillation_phase=oscillation_phase)
            layer_outputs.append(h)
            layer_metadata.append(meta)
        
        # Iterative refinement with top-down predictions
        for _ in range(num_iterations - 1):
            for i, layer in enumerate(self.layers[:-1]):
                pred = self.predictors[i](layer_outputs[i + 2])
                h, meta = layer(layer_outputs[i + 1], top_down=pred,
                               oscillation_phase=oscillation_phase)
                layer_outputs[i + 1] = h
        
        output = self.output_norm(layer_outputs[-1])
        
        return output, {'layers': layer_metadata}
