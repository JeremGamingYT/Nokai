"""
Hippocampal Memory - External memory system for long-term storage

Uses vector databases (FAISS/HNSWLIB) for efficient similarity search,
mimicking the hippocampus's role in memory storage and retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class PatternSeparator(nn.Module):
    """
    Dentate Gyrus - Pattern Separation
    
    Creates orthogonal representations to minimize interference
    between similar memories.
    """
    
    def __init__(self, input_dim: int, expansion_factor: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.expanded_dim = int(input_dim * expansion_factor)
        
        # Expansion layer (sparse random projection)
        self.expansion = nn.Linear(input_dim, self.expanded_dim, bias=False)
        nn.init.sparse_(self.expansion.weight, sparsity=0.9)
        
        # Competitive inhibition
        self.inhibition = nn.Linear(self.expanded_dim, self.expanded_dim, bias=False)
        nn.init.eye_(self.inhibition.weight)
        self.inhibition.weight.data *= -0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand to high-dimensional sparse space
        expanded = F.relu(self.expansion(x))
        
        # Winner-take-all sparsification
        k = max(1, self.expanded_dim // 20)  # 5% active
        values, indices = torch.topk(expanded, k, dim=-1)
        
        sparse_output = torch.zeros_like(expanded)
        sparse_output.scatter_(-1, indices, values)
        
        return sparse_output


class MemoryIndex:
    """Vector database wrapper for memory storage."""
    
    def __init__(self, dim: int, max_size: int = 100000, backend: str = "hnsw"):
        self.dim = dim
        self.max_size = max_size
        self.backend = backend
        self.current_size = 0
        
        if backend == "hnsw" and HNSWLIB_AVAILABLE:
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.init_index(max_elements=max_size, ef_construction=200, M=16)
            self.index.set_ef(50)
        elif backend == "faiss" and FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            # Fallback to simple tensor storage
            self.index = None
            self.storage = torch.zeros(max_size, dim)
    
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Add vectors to the index with capacity check."""
        n = len(vectors)
        
        # Check if we would exceed capacity
        if self.current_size + n > self.max_size:
            # Skip if full (circular buffer would require rebuilding index)
            remaining = self.max_size - self.current_size
            if remaining <= 0:
                return  # Memory is full, skip storing
            # Only add what we can fit
            vectors = vectors[:remaining]
            n = remaining
        
        if ids is None:
            ids = np.arange(self.current_size, self.current_size + n)
        else:
            ids = ids[:n]
        
        if self.index is not None:
            if self.backend == "hnsw":
                self.index.add_items(vectors, ids)
            else:
                self.index.add(vectors)
        else:
            self.storage[self.current_size:self.current_size + n] = torch.from_numpy(vectors)
        
        self.current_size += n
    
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if self.index is not None:
            if self.backend == "hnsw":
                ids, distances = self.index.knn_query(query, k=k)
                return ids, 1 - distances  # Convert distance to similarity
            else:
                distances, ids = self.index.search(query, k)
                return ids, distances
        else:
            # Fallback: brute force
            query_t = torch.from_numpy(query)
            similarities = F.cosine_similarity(
                query_t.unsqueeze(1),
                self.storage[:self.current_size].unsqueeze(0),
                dim=-1
            )
            values, indices = torch.topk(similarities, k, dim=-1)
            return indices.numpy(), values.numpy()


class HippocampalMemory(nn.Module):
    """
    Complete hippocampal memory system.
    
    Implements:
    - Pattern separation (DG)
    - Auto-associative memory (CA3)
    - Output mapping (CA1)
    - External vector storage
    """
    
    def __init__(
        self,
        embedding_dim: int,
        memory_size: int = 100000,
        num_heads: int = 4,
        retrieval_top_k: int = 5,
        backend: str = "hnsw",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.retrieval_top_k = retrieval_top_k
        
        # Pattern Separation (Dentate Gyrus)
        self.pattern_separator = PatternSeparator(embedding_dim)
        
        # Encoder for memory keys
        self.key_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Encoder for memory values
        self.value_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # CA3 - Auto-associative network
        self.ca3_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # CA1 - Output projection
        self.ca1_output = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
        # External memory index
        self.memory_index = MemoryIndex(embedding_dim, memory_size, backend)
        
        # Value storage (parallel to index)
        self.register_buffer('memory_values', torch.zeros(memory_size, embedding_dim))
        self.memory_count = 0
    
    def store(self, keys: torch.Tensor, values: torch.Tensor):
        """Store key-value pairs in memory."""
        # Encode keys and values
        encoded_keys = self.key_encoder(keys)
        encoded_values = self.value_encoder(values)
        
        # Apply pattern separation
        separated_keys = self.pattern_separator(encoded_keys)
        
        # Add to index
        keys_np = encoded_keys.detach().cpu().numpy()
        self.memory_index.add(keys_np)
        
        # Store values
        n = keys.shape[0]
        if self.memory_count + n <= self.memory_size:
            self.memory_values[self.memory_count:self.memory_count + n] = encoded_values.detach()
            self.memory_count += n
    
    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve relevant memories for query."""
        batch_size = query.shape[0]
        
        # Encode query
        encoded_query = self.key_encoder(query)
        
        if self.memory_count == 0:
            return torch.zeros_like(query), torch.zeros(batch_size, self.retrieval_top_k)
        
        # Search in memory
        query_np = encoded_query.detach().cpu().numpy()
        indices, similarities = self.memory_index.search(query_np, k=self.retrieval_top_k)
        
        # Retrieve values
        indices_t = torch.from_numpy(indices).long().to(query.device)
        retrieved_values = self.memory_values[indices_t.flatten()].view(
            batch_size, self.retrieval_top_k, -1
        )
        
        # Apply CA3 attention for pattern completion
        completed, _ = self.ca3_attention(
            encoded_query.unsqueeze(1),
            retrieved_values,
            retrieved_values,
        )
        
        # CA1 output
        output = self.ca1_output(completed.squeeze(1))
        
        return output, torch.from_numpy(similarities).to(query.device)
    
    def forward(
        self,
        x: torch.Tensor,
        store: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through hippocampal memory.
        
        Args:
            x: Input tensor [batch, seq, dim] or [batch, dim]
            store: Whether to store this input in memory
        """
        is_sequence = x.dim() == 3
        if is_sequence:
            batch, seq, dim = x.shape
            x_flat = x.view(-1, dim)
        else:
            x_flat = x
        
        # Retrieve relevant memories
        retrieved, similarities = self.retrieve(x_flat)
        
        # Combine with input
        output = x_flat + 0.5 * retrieved
        
        # Optionally store
        if store and self.training:
            self.store(x_flat.detach(), x_flat.detach())
        
        if is_sequence:
            output = output.view(batch, seq, dim)
        
        metadata = {
            'memories_retrieved': self.retrieval_top_k,
            'memory_count': self.memory_count,
            'mean_similarity': similarities.mean().item() if similarities.numel() > 0 else 0,
        }
        
        return output, metadata
