"""
Cortical Layer - Ensemble of cortical columns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from nokai.config import CorticalColumnConfig, ThalamusConfig
from nokai.cortex.column import CorticalColumn


class ColumnRouter(nn.Module):
    """Routes input to relevant cortical columns with sparse selection."""
    
    def __init__(self, input_dim: int, num_columns: int, sparsity: float = 0.05):
        super().__init__()
        self.num_columns = num_columns
        self.sparsity = sparsity
        self.column_embeddings = nn.Parameter(torch.randn(num_columns, input_dim) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.GELU(),
            nn.Linear(input_dim // 2, num_columns), nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        similarity = torch.matmul(
            F.normalize(x, dim=-1),
            F.normalize(self.column_embeddings, dim=-1).T
        )
        gate_values = self.gate(x)
        logits = similarity + gate_values
        k = max(1, int(self.num_columns * self.sparsity))
        _, top_indices = torch.topk(logits, k, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, top_indices, 1.0)
        weights = weights * F.softmax(logits, dim=-1)
        return weights, (weights > 0).float()


class CorticalLayer(nn.Module):
    """Layer of cortical columns with lateral connections."""
    
    def __init__(self, num_columns: int, column_config: CorticalColumnConfig,
                 input_dim: int, thalamus_config: Optional[ThalamusConfig] = None):
        super().__init__()
        self.num_columns = num_columns
        self.column_config = column_config
        self.input_dim = input_dim
        
        self.columns = nn.ModuleList([
            CorticalColumn(column_config, i, input_dim, column_config.num_neurons)
            for i in range(num_columns)
        ])
        
        sparsity = thalamus_config.sparsity_target if thalamus_config else 0.05
        self.router = ColumnRouter(input_dim, num_columns, sparsity)
        self.output_proj = nn.Linear(column_config.num_neurons * num_columns, input_dim)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor, top_down: Optional[torch.Tensor] = None,
                oscillation_phase: Optional[float] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = x.shape
        routing_weights, _ = self.router(x)
        
        column_outputs = []
        for i, column in enumerate(self.columns):
            col_weight = routing_weights[:, :, i:i+1]
            if col_weight.max() > 0.01:
                col_input = x.view(-1, self.input_dim)
                col_out, _ = column(col_input, oscillation_phase=oscillation_phase)
                col_out = col_out.view(batch_size, seq_len, -1) * col_weight
            else:
                col_out = torch.zeros(batch_size, seq_len, self.column_config.num_neurons,
                                      device=x.device, dtype=x.dtype)
            column_outputs.append(col_out)
        
        combined = torch.cat(column_outputs, dim=-1)
        output = self.norm(self.output_proj(combined) + x)
        
        return output, {'active_columns': (routing_weights > 0.01).sum(-1).float().mean().item()}
