"""
Nōkai Main Model - Complete brain architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from nokai.config import NokaiConfig
from nokai.cortex import Cortex
from nokai.hippocampus import HippocampalMemory
from nokai.oscillations import OscillatorNetwork


class NokaiEmbedding(nn.Module):
    """Token and positional embeddings with optional sparse encoding."""
    
    def __init__(self, config: NokaiConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Learned positional embeddings
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        
        # Projection if needed
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.column_config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(positions)
        
        # Combine
        x = tok_emb + pos_emb
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class NokaiOutput(nn.Module):
    """Output head for language modeling."""
    
    def __init__(self, config: NokaiConfig, embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        
        # Optional weight tying
        if config.shared_embeddings and embedding is not None:
            self.output_projection = None
            self.embedding = embedding
        else:
            self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
            self.embedding = None
        
        self.norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        
        if self.output_projection is not None:
            logits = self.output_projection(x)
        else:
            logits = F.linear(x, self.embedding.weight)
        
        return logits


class NokaiModel(nn.Module):
    """
    Nōkai - Bio-inspired artificial brain for language understanding.
    
    Components:
    - Embedding: Converts tokens to continuous representations
    - Cortex: Hierarchical processing with cortical columns
    - Hippocampus: External memory for long-term storage
    - Oscillations: Neural synchronization for coordination
    - Output: Language modeling head
    """
    
    def __init__(self, config: NokaiConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = NokaiEmbedding(config)
        
        # Cortical processing
        self.cortex = Cortex(config)
        
        # Hippocampal memory (optional)
        if config.hippocampus.enabled:
            self.hippocampus = HippocampalMemory(
                embedding_dim=config.embedding_dim,
                memory_size=config.hippocampus.memory_size,
                num_heads=config.hippocampus.num_heads_ca3,
                retrieval_top_k=config.hippocampus.retrieval_top_k,
            )
        else:
            self.hippocampus = None
        
        # Neural oscillations (optional)
        if config.oscillations.enabled:
            self.oscillations = OscillatorNetwork(
                num_oscillators=config.num_columns,
                theta_freq=config.oscillations.theta_freq,
                gamma_freq=config.oscillations.gamma_freq,
                coupling_strength=config.oscillations.coupling_strength,
            )
        else:
            self.oscillations = None
        
        # Output head
        self.output = NokaiOutput(
            config, 
            self.embedding.token_embedding if config.shared_embeddings else None
        )
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.memory_optimization.gradient_checkpointing
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print parameter count
        self._print_params()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Nōkai Model initialized:")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Estimated VRAM: {self.config.estimate_vram_mb():.1f} MB")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        store_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Nōkai.
        
        Args:
            input_ids: Token IDs [batch, seq]
            labels: Target labels for loss computation
            store_memory: Whether to store in hippocampal memory
            
        Returns:
            Dict with logits, loss, and metadata
        """
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Update oscillations
        oscillation_phase = None
        if self.oscillations is not None:
            self.oscillations.step()
            oscillation_phase = self.oscillations.theta_phases.mean().item()
        
        # Cortical processing
        if self.gradient_checkpointing and self.training:
            x, cortex_meta = torch.utils.checkpoint.checkpoint(
                self.cortex, x, oscillation_phase,
                use_reentrant=False
            )
        else:
            x, cortex_meta = self.cortex(x, oscillation_phase=oscillation_phase)
        
        # Hippocampal memory
        hippocampus_meta = {}
        if self.hippocampus is not None:
            x, hippocampus_meta = self.hippocampus(x, store=store_memory)
        
        # Output logits
        logits = self.output(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'cortex_metadata': cortex_meta,
            'hippocampus_metadata': hippocampus_meta,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop if too long
            if input_ids.shape[1] > self.config.max_sequence_length:
                input_ids = input_ids[:, -self.config.max_sequence_length:]
            
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def consolidate_memory(self, num_replays: int = 10):
        """
        Trigger memory consolidation (sleep phase).
        
        Biological Parallel:
            During sleep, the hippocampus "replays" recent memories,
            allowing them to be consolidated into neocortical storage.
            
        Args:
            num_replays: Number of memory replay iterations
        """
        if self.hippocampus is not None and self.hippocampus.memory_count > 0:
            # Replay stored memories through the network
            with torch.no_grad():
                for _ in range(num_replays):
                    # Sample random memories from hippocampus
                    num_to_sample = min(8, self.hippocampus.memory_count)
                    indices = torch.randint(
                        0, self.hippocampus.memory_count, 
                        (num_to_sample,),
                        device=self.hippocampus.memory_values.device
                    )
                    
                    # Retrieve stored memories
                    memories = self.hippocampus.memory_values[indices]
                    
                    # Replay through cortex (lightweight forward pass)
                    if memories.numel() > 0:
                        replayed, _ = self.cortex(memories.unsqueeze(0))
                        
                        # The replay strengthens synaptic connections
                        # (This happens automatically via the cortex's internal state)
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get statistics on activation and weight sparsity."""
        stats = {}
        
        # Weight sparsity
        total_weights = 0
        zero_weights = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                zero_weights += (param.abs() < 1e-6).sum().item()
        
        stats['weight_sparsity'] = zero_weights / total_weights if total_weights > 0 else 0
        
        return stats
