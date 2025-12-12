"""
Ternary Weights - Native {-1, 0, +1} Quantization

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

Synapses in the brain operate in essentially three modes:
- Excitatory (+1): Release glutamate, depolarize postsynaptic neuron
- Inhibitory (-1): Release GABA, hyperpolarize postsynaptic neuron  
- Silent (0): Synapse exists but is currently inactive

This ternary nature is fundamentally different from float32 precision!

=============================================================================
EFFICIENCY GAINS
=============================================================================

Storage:
- float32: 32 bits per weight
- float16: 16 bits per weight
- ternary: 2 bits per weight (16x compression!)

Compute:
- float matmul: FMA (Fused Multiply-Add)
- ternary matmul: XOR + POPCOUNT (5-10x faster on specialized hardware)

Energy:
- float ops: ~4.6 pJ per operation
- ternary ops: ~0.1 pJ per operation (46x more efficient!)

=============================================================================
IMPLEMENTATION: Straight-Through Estimator (STE)
=============================================================================

The challenge: ternary quantization is non-differentiable!

Solution: Use continuous "latent weights" during training, 
quantize only during forward pass, and pass gradients through unchanged.

Forward:  w_ternary = sign(w_latent) * (|w_latent| > threshold)
Backward: ∂L/∂w_latent = ∂L/∂w_ternary (straight-through)

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Tuple
import math


class TernaryQuantize(Function):
    """
    Ternary quantization with Straight-Through Estimator.
    
    Forward: w → {-1, 0, +1}
    Backward: gradient passes through unchanged
    """
    
    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """
        Quantize weights to ternary values.
        
        Args:
            weight: Latent weight tensor
            threshold: Values with |w| < threshold become 0
            
        Returns:
            Ternary weight ∈ {-1, 0, +1}
        """
        ctx.save_for_backward(weight)
        ctx.threshold = threshold
        
        # Create ternary mask
        # |w| > threshold → ±1, else → 0
        mask = (weight.abs() > threshold).float()
        ternary = torch.sign(weight) * mask
        
        return ternary
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Straight-Through Estimator: pass gradient unchanged.
        
        With optional gradient clipping for stability.
        """
        weight, = ctx.saved_tensors
        
        # STE: gradient flows through unchanged
        grad_weight = grad_output.clone()
        
        # Optional: clip gradients for weights far from threshold
        # This helps training stability
        clip_mask = (weight.abs() < 2.0).float()
        grad_weight = grad_weight * clip_mask
        
        return grad_weight, None


def ternary_quantize(weight: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Functional interface for ternary quantization."""
    return TernaryQuantize.apply(weight, threshold)


class TernaryLinear(nn.Module):
    """
    Linear layer with native ternary weights.
    
    ═══════════════════════════════════════════════════════════════════════
    USAGE
    ═══════════════════════════════════════════════════════════════════════
    
    layer = TernaryLinear(256, 512)
    output = layer(input)  # Uses ternary weights in forward
    loss.backward()        # Gradients update latent weights
    
    ═══════════════════════════════════════════════════════════════════════
    COMPUTATIONAL ADVANTAGE
    ═══════════════════════════════════════════════════════════════════════
    
    Standard matmul: y = Wx
        → For each output: sum(w_i * x_i) with float multiply-add
        
    Ternary matmul: y = W_ternary @ x
        → For each output: sum(x_i where w=+1) - sum(x_i where w=-1)
        → No multiplication! Just additions and subtractions!
        
    On specialized hardware (FPGA, neuromorphic chips):
        - 10x fewer operations
        - 10x less memory bandwidth
        - 10x less energy
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.05,
        init_scale: float = 1.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        # Latent weights (continuous, trained via STE)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Statistics tracking
        self.register_buffer('sparsity', torch.tensor(0.0))
        self.register_buffer('ternary_ratio', torch.tensor(0.0))
        
        # Initialize
        self._init_weights(init_scale)
    
    def _init_weights(self, scale: float):
        """
        Initialize latent weights for good ternary coverage.
        
        Goal: ~60% of weights should be non-zero after ternarization.
        """
        # Kaiming-like but scaled for ternary
        std = scale * math.sqrt(2.0 / self.in_features)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        
        # Shift distribution so threshold creates ~60% non-zero
        # (Empirically tuned)
        self.weight.data *= 2.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ternary weights.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Quantize to ternary
        w_ternary = ternary_quantize(self.weight, self.threshold)
        
        # Track statistics
        if self.training:
            with torch.no_grad():
                zeros = (w_ternary == 0).float().mean()
                self.sparsity = 0.99 * self.sparsity + 0.01 * zeros
                
                # Ratio of +1 to -1
                pos = (w_ternary == 1).float().sum()
                neg = (w_ternary == -1).float().sum()
                ratio = pos / (neg + 1e-8)
                self.ternary_ratio = 0.99 * self.ternary_ratio + 0.01 * ratio
        
        # Linear operation
        output = F.linear(x, w_ternary, self.bias)
        
        return output
    
    def get_ternary_weights(self) -> torch.Tensor:
        """Get current ternary weights (no gradient)."""
        with torch.no_grad():
            return ternary_quantize(self.weight, self.threshold)
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        w = self.get_ternary_weights()
        total = w.numel()
        zeros = (w == 0).sum().item()
        pos = (w == 1).sum().item()
        neg = (w == -1).sum().item()
        
        return {
            'total_weights': total,
            'zeros': zeros,
            'positives': pos,
            'negatives': neg,
            'sparsity': zeros / total,
            'compression_ratio': 16.0,  # vs float32
            'effective_bits': 2.0,
            'original_bits': 32.0,
        }
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'threshold={self.threshold}'
        )


class TernaryConv2d(nn.Module):
    """
    2D Convolution with ternary weights.
    
    Same STE approach as TernaryLinear, but for conv operations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold: float = 0.05,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        
        # Latent weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = math.sqrt(2.0 / fan_in) * 2.0
        nn.init.normal_(self.weight, mean=0.0, std=std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary = ternary_quantize(self.weight, self.threshold)
        return F.conv2d(x, w_ternary, self.bias, self.stride, self.padding)


class TernaryEmbedding(nn.Module):
    """
    Embedding layer with ternary sparse vectors.
    
    Each token is represented by a sparse ternary vector.
    This enables extremely efficient similarity computation.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sparsity: float = 0.9,
        threshold: float = 0.05,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity = sparsity
        self.threshold = threshold
        
        # Latent embeddings
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with controlled sparsity
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        
        # Scale to achieve target sparsity after ternarization
        # Higher threshold → more zeros → more sparse
        scale = 1.0 / (1.0 - self.sparsity + 0.1)
        self.weight.data *= scale
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Get latent embeddings
        embeddings = F.embedding(indices, self.weight)
        
        # Quantize to ternary
        ternary_embeddings = ternary_quantize(embeddings, self.threshold)
        
        return ternary_embeddings
    
    def similarity(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient ternary similarity search.
        
        With ternary weights, dot product becomes:
        count(query[w=+1]) - count(query[w=-1])
        """
        # Quantize query
        q_ternary = ternary_quantize(query, self.threshold)
        
        # Get all ternary embeddings
        all_embeddings = ternary_quantize(self.weight, self.threshold)
        
        # Ternary dot product (no multiplication!)
        # This could be optimized with bit operations on hardware
        scores = torch.matmul(q_ternary, all_embeddings.T)
        
        # Top-k
        top_scores, top_indices = torch.topk(scores, top_k, dim=-1)
        
        return top_scores, top_indices


class ScaledTernaryLinear(nn.Module):
    """
    Ternary Linear with learned scale factor (à la BitNet b1.58).
    
    y = α * (W_ternary @ x)
    
    Where α is a learned per-output-channel scale.
    This allows more expressivity while keeping weights ternary.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.05,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        # Ternary weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Learned scale per output channel
        self.scale = nn.Parameter(torch.ones(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        std = math.sqrt(2.0 / self.in_features) * 2.0
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.ones_(self.scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary = ternary_quantize(self.weight, self.threshold)
        
        # Compute ternary linear
        out = F.linear(x, w_ternary)
        
        # Apply scale
        out = out * self.scale
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_ternary_params(model: nn.Module) -> dict:
    """Count ternary vs float parameters in a model."""
    ternary_params = 0
    float_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (TernaryLinear, TernaryConv2d, TernaryEmbedding, ScaledTernaryLinear)):
            ternary_params += module.weight.numel()
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            float_params += sum(p.numel() for p in module.parameters())
    
    return {
        'ternary_params': ternary_params,
        'float_params': float_params,
        'ternary_ratio': ternary_params / (ternary_params + float_params + 1e-8),
        'ternary_storage_mb': ternary_params * 2 / 8 / 1024 / 1024,  # 2 bits
        'float_storage_mb': float_params * 32 / 8 / 1024 / 1024,  # 32 bits
    }


def convert_to_ternary(model: nn.Module, threshold: float = 0.05) -> nn.Module:
    """
    Convert all Linear layers in a model to TernaryLinear.
    
    Warning: This creates new parameters, so optimizer must be recreated!
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            ternary = TernaryLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                threshold=threshold,
            )
            # Copy weights
            ternary.weight.data = module.weight.data.clone()
            if module.bias is not None:
                ternary.bias.data = module.bias.data.clone()
            
            setattr(model, name, ternary)
        else:
            convert_to_ternary(module, threshold)
    
    return model
