"""
Nōkai Configuration Module

Defines all hyperparameters and architectural choices for the Nōkai model.
Configurations are validated using Pydantic for type safety.
"""

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    """Type of external memory storage."""
    INTERNAL = "internal"      # GPU tensor
    FAISS = "faiss"           # FAISS index
    HNSW = "hnsw"             # HNSW graph
    MMAP = "mmap"             # Memory-mapped file


class AttentionType(str, Enum):
    """Type of attention mechanism."""
    FULL = "full"                     # Standard O(n²)
    SPARSE_THALAMIC = "sparse_thalamic"  # Our bio-inspired O(n log n)
    LINEAR = "linear"                 # Linear attention
    FLASH = "flash"                   # Flash Attention 2


class LearningRule(str, Enum):
    """Learning rule for weight updates."""
    BACKPROP = "backprop"             # Standard backpropagation
    HEBBIAN_STDP = "hebbian_stdp"     # Spike-timing dependent plasticity
    PREDICTIVE = "predictive"          # Predictive coding
    HYBRID = "hybrid"                  # Combination


class CorticalColumnConfig(BaseModel):
    """Configuration for a single cortical column."""
    num_neurons: int = Field(default=256, ge=32, le=4096)
    num_layers: int = Field(default=6, ge=1, le=24)
    activation: str = Field(default="gelu")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    lateral_connectivity: float = Field(default=0.1, ge=0.0, le=1.0)
    

class HippocampusConfig(BaseModel):
    """Configuration for hippocampal memory system."""
    enabled: bool = True
    memory_size: int = Field(default=100_000, ge=1000)
    embedding_dim: int = Field(default=256, ge=64)
    num_heads_ca3: int = Field(default=4, ge=1)
    pattern_separation_factor: float = Field(default=10.0, ge=1.0)
    retrieval_top_k: int = Field(default=5, ge=1)
    memory_type: MemoryType = MemoryType.HNSW


class ThalamusConfig(BaseModel):
    """Configuration for thalamic attention routing."""
    num_clusters: int = Field(default=64, ge=8)
    routing_temperature: float = Field(default=1.0, ge=0.1)
    sparsity_target: float = Field(default=0.05, ge=0.01, le=0.5)
    oscillation_coupling: float = Field(default=0.3, ge=0.0, le=1.0)


class CerebellumConfig(BaseModel):
    """Configuration for cerebellar timing module."""
    enabled: bool = True
    num_granule_cells: int = Field(default=1024, ge=128)
    num_purkinje_cells: int = Field(default=256, ge=32)
    timing_resolution_ms: float = Field(default=10.0, ge=1.0)


class OscillationConfig(BaseModel):
    """Configuration for neural oscillations."""
    enabled: bool = True
    theta_freq: float = Field(default=6.0, ge=4.0, le=8.0)
    gamma_freq: float = Field(default=40.0, ge=30.0, le=100.0)
    coupling_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    phase_encoding: bool = True


class LearningConfig(BaseModel):
    """Configuration for learning mechanisms."""
    rule: LearningRule = LearningRule.HYBRID
    learning_rate: float = Field(default=1e-4, ge=1e-7, le=1.0)
    hebbian_lr: float = Field(default=1e-3, ge=1e-6, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_clip: float = Field(default=1.0, ge=0.0)
    plasticity_modulation: bool = True
    sleep_consolidation: bool = True
    consolidation_interval: int = Field(default=1000, ge=100)


class MemoryOptimizationConfig(BaseModel):
    """Configuration for memory optimizations."""
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    sparse_gradients: bool = True
    offload_optimizer: bool = False
    pin_memory: bool = True
    prefetch_factor: int = Field(default=2, ge=1)


class NokaiConfig(BaseModel):
    """
    Master configuration for Nōkai model.
    
    Preset configurations:
        - nano: ~4M params, 200MB VRAM
        - micro: ~17M params, 500MB VRAM
        - mini: ~67M params, 2GB VRAM
        - base: ~268M params, 6GB VRAM (RTX 5070 target)
        - large: ~1B params, 16GB VRAM
    """
    
    # Model architecture
    num_columns: int = Field(default=1024, ge=64, le=16384)
    column_config: CorticalColumnConfig = Field(default_factory=CorticalColumnConfig)
    
    # Vocabulary and embeddings
    vocab_size: int = Field(default=32000, ge=1000)
    max_sequence_length: int = Field(default=4096, ge=128)
    embedding_dim: int = Field(default=256, ge=64)
    shared_embeddings: bool = True
    
    # Brain regions
    hippocampus: HippocampusConfig = Field(default_factory=HippocampusConfig)
    thalamus: ThalamusConfig = Field(default_factory=ThalamusConfig)
    cerebellum: CerebellumConfig = Field(default_factory=CerebellumConfig)
    oscillations: OscillationConfig = Field(default_factory=OscillationConfig)
    
    # Attention
    attention_type: AttentionType = AttentionType.SPARSE_THALAMIC
    num_attention_heads: int = Field(default=8, ge=1)
    
    # Learning
    learning: LearningConfig = Field(default_factory=LearningConfig)
    
    # Memory optimization
    memory_optimization: MemoryOptimizationConfig = Field(default_factory=MemoryOptimizationConfig)
    
    # Hardware
    device: str = "cuda"
    dtype: str = "float16"
    compile_model: bool = True  # torch.compile()
    
    @classmethod
    def nano(cls) -> "NokaiConfig":
        """~4M parameters, 200MB VRAM."""
        return cls(
            num_columns=256,
            column_config=CorticalColumnConfig(num_neurons=64, num_layers=4),
            embedding_dim=128,
            hippocampus=HippocampusConfig(memory_size=10_000, embedding_dim=128),
            thalamus=ThalamusConfig(num_clusters=16),
        )
    
    @classmethod
    def micro(cls) -> "NokaiConfig":
        """~17M parameters, 500MB VRAM."""
        return cls(
            num_columns=512,
            column_config=CorticalColumnConfig(num_neurons=128, num_layers=4),
            embedding_dim=192,
            hippocampus=HippocampusConfig(memory_size=50_000, embedding_dim=192),
            thalamus=ThalamusConfig(num_clusters=32),
        )
    
    @classmethod
    def mini(cls) -> "NokaiConfig":
        """~67M parameters, 2GB VRAM."""
        return cls(
            num_columns=1024,
            column_config=CorticalColumnConfig(num_neurons=256, num_layers=6),
            embedding_dim=256,
            hippocampus=HippocampusConfig(memory_size=100_000, embedding_dim=256),
            thalamus=ThalamusConfig(num_clusters=64),
        )
    
    @classmethod
    def base(cls) -> "NokaiConfig":
        """~268M parameters, 6GB VRAM. Target for RTX 5070."""
        return cls(
            num_columns=2048,
            column_config=CorticalColumnConfig(num_neurons=512, num_layers=8),
            embedding_dim=512,
            hippocampus=HippocampusConfig(memory_size=500_000, embedding_dim=512),
            thalamus=ThalamusConfig(num_clusters=128),
            num_attention_heads=16,
        )
    
    @classmethod
    def large(cls) -> "NokaiConfig":
        """~1B parameters, 16GB VRAM."""
        return cls(
            num_columns=4096,
            column_config=CorticalColumnConfig(num_neurons=1024, num_layers=12),
            embedding_dim=768,
            hippocampus=HippocampusConfig(memory_size=1_000_000, embedding_dim=768),
            thalamus=ThalamusConfig(num_clusters=256),
            num_attention_heads=24,
        )
    
    @classmethod
    def massive(cls) -> "NokaiConfig":
        """
        ~1.8B parameters, 80GB VRAM (H100 target).
        
        100% Biomimetic Architecture:
        - No standard Transformer self-attention
        - Sparse thalamic routing
        - Hebbian plasticity enabled
        - Oscillation-synchronized processing
        
        Parameter distribution:
        - Embeddings: ~102M (50k × 2048)
        - Cortex: ~1.68B (48 layers)
        - Hippocampus: ~20M
        - Thalamus + Limbic: ~8M
        """
        return cls(
            num_columns=8192,
            column_config=CorticalColumnConfig(
                num_neurons=512,
                num_layers=6,
                dropout=0.1,
                lateral_connectivity=0.1,
            ),
            vocab_size=50_000,
            max_sequence_length=4096,
            embedding_dim=2048,
            hippocampus=HippocampusConfig(
                enabled=True,
                memory_size=2_000_000,
                embedding_dim=2048,
                num_heads_ca3=16,
                retrieval_top_k=32,
            ),
            thalamus=ThalamusConfig(
                num_clusters=256,
                sparsity_target=0.05,
                oscillation_coupling=0.6,
            ),
            oscillations=OscillationConfig(
                enabled=True,
                theta_freq=6.0,
                gamma_freq=40.0,
                coupling_strength=0.6,
            ),
            num_attention_heads=32,
            memory_optimization=MemoryOptimizationConfig(
                gradient_checkpointing=True,
                mixed_precision=True,
                activation_checkpointing=True,
            ),
            compile_model=True,
        )
    
    @property
    def hidden_dim(self) -> int:
        """Total hidden dimension across all columns."""
        return self.num_columns * self.column_config.num_neurons
    
    def estimate_parameters(self) -> int:
        """Estimate total parameter count."""
        # Rough estimation
        embedding_params = self.vocab_size * self.embedding_dim * 2  # embed + unembed
        column_params = (
            self.num_columns 
            * self.column_config.num_neurons 
            * self.column_config.num_neurons 
            * self.column_config.num_layers
        )
        attention_params = (
            self.embedding_dim * self.embedding_dim * 4  # Q, K, V, O
            * self.column_config.num_layers
        )
        hippocampus_params = self.hippocampus.embedding_dim ** 2 * 4
        
        total = embedding_params + column_params + attention_params + hippocampus_params
        return int(total)
    
    def estimate_vram_mb(self) -> float:
        """Estimate VRAM usage in MB."""
        params = self.estimate_parameters()
        bytes_per_param = 2 if self.dtype == "float16" else 4
        
        # Parameters + gradients + optimizer states + activations
        param_memory = params * bytes_per_param
        gradient_memory = param_memory if not self.memory_optimization.gradient_checkpointing else param_memory * 0.3
        optimizer_memory = param_memory * 2  # Adam states
        activation_memory = param_memory * 0.5  # Rough estimate
        
        total_bytes = param_memory + gradient_memory + optimizer_memory + activation_memory
        return total_bytes / (1024 * 1024)
