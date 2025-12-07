"""
Semantic Memory - Long-Term Knowledge Storage (Neocortex)

Biological Parallel:
    The neocortex stores semantic knowledge - facts, concepts, and
    general world knowledge. Unlike episodic memory (hippocampus):
    
    1. Updates slowly (sleep consolidation)
    2. Stores abstractions, not specific episodes
    3. Highly compressed and generalized
    4. Organized hierarchically by concept
    
Implementation:
    We implement semantic memory as a slowly-updating knowledge
    base with:
    1. Vector embeddings for concepts
    2. Hierarchical organization
    3. Slow (consolidated) updates
    4. Efficient retrieval via ANN
    
Efficiency:
    - Memory-mapped storage for billion+ entries
    - O(log N) retrieval via approximate nearest neighbor
    - Batch consolidation for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
import mmap
import os
from pathlib import Path

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False


class SemanticMemory(nn.Module):
    """
    Neocortical Semantic Memory - Long-Term Knowledge
    
    Biological Parallel:
        The neocortex gradually acquires general knowledge through
        slow consolidation from hippocampal episodic memory.
        
        Key properties:
        1. Slow learning (avoids catastrophic forgetting)
        2. Abstract representations (concepts, not instances)
        3. Hierarchical organization
        4. Highly efficient retrieval
        
    Implementation:
        - HNSW index for O(log N) retrieval
        - Memory-mapped values for billion-scale storage
        - Slow update rate (consolidated during "sleep")
        - Concept embeddings with hierarchical structure
        
    Efficiency:
        - Scales to billions of concepts
        - O(log N) retrieval
        - Memory-mapped for minimal RAM usage
        - Batch processing for updates
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        max_concepts: int = 1_000_000,
        update_rate: float = 0.001,  # Very slow learning
        storage_path: Optional[str] = None,
        use_mmap: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_concepts = max_concepts
        self.update_rate = update_rate
        self.storage_path = storage_path
        self.use_mmap = use_mmap
        
        # ============================================
        # CONCEPT ENCODER
        # ============================================
        # Encodes inputs into semantic space
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # ============================================
        # CONCEPT DECODER
        # ============================================
        # Reconstructs information from concept
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # ============================================
        # INDEX FOR FAST RETRIEVAL
        # ============================================
        if HNSWLIB_AVAILABLE:
            self.index = hnswlib.Index(space='cosine', dim=embedding_dim)
            self.index.init_index(max_elements=max_concepts, ef_construction=200, M=16)
            self.index.set_ef(50)
        else:
            self.index = None
        
        # ============================================
        # CONCEPT STORAGE
        # ============================================
        if use_mmap and storage_path:
            self._init_mmap_storage(storage_path)
        else:
            # In-memory storage
            self.register_buffer('concepts', torch.zeros(max_concepts, embedding_dim))
            self.register_buffer('concept_counts', torch.zeros(max_concepts))  # Access frequency
            self.mmap_file = None
        
        # Metadata
        self.register_buffer('num_concepts', torch.tensor(0, dtype=torch.long))
        
        # ============================================
        # SYNAPTIC WEIGHTS (Plasticity)
        # ============================================
        # Concepts that are frequently retrieved strengthen
        self.register_buffer('synaptic_weights', torch.ones(max_concepts))
        
        # ============================================
        # CONSOLIDATION BUFFER
        # ============================================
        # Buffer for experiences waiting to be consolidated
        self.consolidation_buffer: List[torch.Tensor] = []
        self.max_buffer_size = 1000
    
    def _init_mmap_storage(self, storage_path: str):
        """Initialize memory-mapped storage for large-scale data."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        concept_file = self.storage_path / "concepts.dat"
        
        # Calculate file size
        file_size = self.max_concepts * self.embedding_dim * 4  # float32
        
        if not concept_file.exists():
            # Create the file
            with open(concept_file, "wb") as f:
                f.write(b'\x00' * file_size)
        
        # Memory-map the file
        self.mmap_file = open(concept_file, "r+b")
        self.mmap_buffer = mmap.mmap(self.mmap_file.fileno(), file_size)
        
        # Create numpy array view
        self.concepts = np.frombuffer(self.mmap_buffer, dtype=np.float32)
        self.concepts = self.concepts.reshape(self.max_concepts, self.embedding_dim)
    
    def store(
        self,
        experience: torch.Tensor,
        immediate: bool = False,
    ):
        """
        Store experience for later consolidation.
        
        Biological Parallel:
            Experiences are first stored temporarily (hippocampus)
            and later consolidated to neocortex during "sleep".
            
        Args:
            experience: Experience embedding [dim] or [batch, dim]
            immediate: Force immediate storage (bypass consolidation)
        """
        if immediate:
            self._immediate_store(experience)
        else:
            # Add to consolidation buffer
            if experience.dim() == 1:
                experience = experience.unsqueeze(0)
            
            for i in range(experience.shape[0]):
                if len(self.consolidation_buffer) < self.max_buffer_size:
                    self.consolidation_buffer.append(experience[i].detach().clone())
    
    def _immediate_store(self, concept: torch.Tensor):
        """Immediately store a concept."""
        if concept.dim() == 2:
            concept = concept.mean(0)  # Aggregate
        
        # Encode into semantic space
        encoded = self.encoder(concept.unsqueeze(0)).squeeze(0)
        encoded_np = encoded.detach().cpu().numpy()
        
        # Check for similar existing concept
        if self.num_concepts > 0 and self.index is not None:
            ids, distances = self.index.knn_query(encoded_np.reshape(1, -1), k=1)
            
            if distances[0, 0] < 0.1:  # Similar concept exists
                # Update existing (slow integration)
                existing_id = ids[0, 0]
                
                if self.use_mmap:
                    existing = torch.from_numpy(
                        self.concepts[existing_id].copy()
                    ).to(concept.device)
                else:
                    existing = self.concepts[existing_id]
                
                # Slow weighted update
                updated = (1 - self.update_rate) * existing + self.update_rate * encoded
                
                if self.use_mmap:
                    self.concepts[existing_id] = updated.detach().cpu().numpy()
                else:
                    self.concepts[existing_id] = updated.detach()
                
                self.concept_counts[existing_id] += 1
                return existing_id
        
        # Store new concept
        if self.num_concepts < self.max_concepts:
            idx = self.num_concepts.item()
            
            if self.use_mmap:
                self.concepts[idx] = encoded_np
            else:
                self.concepts[idx] = encoded.detach()
            
            if self.index is not None:
                self.index.add_items(encoded_np.reshape(1, -1), np.array([idx]))
            
            self.num_concepts += 1
            self.concept_counts[idx] = 1
            
            return idx
        
        return -1  # Memory full
    
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant concepts from semantic memory.
        
        Args:
            query: Query embedding [batch, dim] or [dim]
            k: Number of concepts to retrieve
            
        Returns:
            concepts: Retrieved concept embeddings
            similarities: Similarity scores
        """
        if self.num_concepts == 0:
            return torch.zeros_like(query), torch.zeros(1)
        
        single_query = query.dim() == 1
        if single_query:
            query = query.unsqueeze(0)
        
        # Encode query
        encoded_query = self.encoder(query)
        query_np = encoded_query.detach().cpu().numpy()
        
        # Retrieve from index
        k = min(k, self.num_concepts.item())
        
        if self.index is not None:
            ids, distances = self.index.knn_query(query_np, k=k)
            similarities = 1 - distances  # Convert distance to similarity
        else:
            # Fallback: brute force
            if self.use_mmap:
                all_concepts = torch.from_numpy(
                    self.concepts[:self.num_concepts.item()].copy()
                ).to(query.device)
            else:
                all_concepts = self.concepts[:self.num_concepts.item()]
            
            sims = F.cosine_similarity(
                encoded_query.unsqueeze(1),
                all_concepts.unsqueeze(0),
                dim=-1
            )
            similarities, ids = torch.topk(sims, k, dim=-1)
            ids = ids.cpu().numpy()
            similarities = similarities.cpu().numpy()
        
        # Gather concepts
        batch_size = query.shape[0]
        retrieved = []
        
        for b in range(batch_size):
            if self.use_mmap:
                concepts_b = torch.from_numpy(
                    self.concepts[ids[b]].copy()
                ).to(query.device)
            else:
                concepts_b = self.concepts[ids[b]]
            retrieved.append(concepts_b)
            
            # Update synaptic weights (frequently accessed = stronger)
            for idx in ids[b]:
                if idx < len(self.synaptic_weights):
                    self.synaptic_weights[idx] = min(
                        self.synaptic_weights[idx] * 1.001,
                        2.0
                    )
        
        retrieved = torch.stack(retrieved)  # [batch, k, dim]
        similarities = torch.from_numpy(similarities).to(query.device)
        
        # Decode retrieved concepts
        decoded = self.decoder(retrieved)
        
        if single_query:
            decoded = decoded.squeeze(0)
            similarities = similarities.squeeze(0)
        
        return decoded, similarities
    
    def consolidate(self, batch_size: int = 32):
        """
        Consolidate buffered experiences (sleep phase).
        
        Biological Parallel:
            During sleep, the hippocampus "replays" recent experiences
            to the neocortex, which slowly integrates them into
            semantic memory.
        """
        if len(self.consolidation_buffer) == 0:
            return 0
        
        # Process in batches
        num_consolidated = 0
        
        while len(self.consolidation_buffer) > 0 and num_consolidated < batch_size:
            experience = self.consolidation_buffer.pop(0)
            self._immediate_store(experience)
            num_consolidated += 1
        
        return num_consolidated
    
    def forward(
        self,
        x: torch.Tensor,
        store: bool = False,
        retrieve: bool = True,
        k: int = 5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through semantic memory.
        
        Args:
            x: Input tensor [batch, seq, dim] or [batch, dim]
            store: Store in consolidation buffer
            retrieve: Retrieve related concepts
            k: Number of concepts to retrieve
            
        Returns:
            output: Memory-augmented output
            metadata: Processing details
        """
        is_seq = x.dim() == 3
        if is_seq:
            batch, seq, dim = x.shape
            x_flat = x.view(-1, dim)
        else:
            x_flat = x
        
        output = x_flat.clone()
        
        # Store if requested
        if store and self.training:
            self.store(x_flat.detach())
        
        # Retrieve if requested and have content
        if retrieve and self.num_concepts > 0:
            retrieved, similarities = self.retrieve(x_flat, k)
            
            # Aggregate retrieved knowledge
            if retrieved.dim() == 3:
                # Weight by similarity
                weights = F.softmax(similarities, dim=-1)
                retrieved_agg = torch.einsum('bk,bkd->bd', weights, retrieved)
            else:
                retrieved_agg = retrieved
            
            # Combine with input (residual)
            output = x_flat + 0.3 * retrieved_agg
        else:
            similarities = torch.zeros(1)
        
        if is_seq:
            output = output.view(batch, seq, dim)
        
        metadata = {
            'num_concepts': self.num_concepts.item(),
            'buffer_size': len(self.consolidation_buffer),
            'mean_similarity': similarities.mean().item() if similarities.numel() > 0 else 0,
            'synaptic_weights_mean': self.synaptic_weights[:max(1, self.num_concepts.item())].mean().item(),
        }
        
        return output, metadata
    
    def energy_check(self) -> bool:
        """Check if semantic retrieval is worthwhile."""
        return self.num_concepts.item() > 10
    
    def __del__(self):
        """Cleanup memory-mapped file."""
        if hasattr(self, 'mmap_buffer') and self.mmap_buffer:
            self.mmap_buffer.close()
        if hasattr(self, 'mmap_file') and self.mmap_file:
            self.mmap_file.close()
