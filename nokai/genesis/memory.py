"""
Modern Hopfield Memory - Exponential Capacity One-Shot Memory

=============================================================================
BIOLOGICAL BACKGROUND
=============================================================================

The hippocampus enables rapid one-shot learning:
- See something once → remember it
- Pattern completion: partial cue → full memory
- Orthogonalization: similar memories stored distinctly

Modern Hopfield Networks (Ramsauer et al., 2020) achieve:
- Exponential capacity: C ~ exp(d/2) patterns (vs d/4 for classical)
- One-iteration retrieval (vs iterative relaxation)
- Continuous values (vs binary)
- Connection to Transformers: softmax attention = Hopfield update!

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class ModernHopfieldMemory(nn.Module):
    """
    Modern Hopfield Network for episodic memory.
    
    ═══════════════════════════════════════════════════════════════════════
    KEY PROPERTIES
    ═══════════════════════════════════════════════════════════════════════
    
    1. EXPONENTIAL CAPACITY:
       - Classical Hopfield: ~0.14 * d patterns
       - Modern Hopfield: ~exp(d/2) patterns!
       
    2. ONE-SHOT STORAGE:
       - Store pattern in O(1)
       - No training required
       
    3. ASSOCIATIVE RETRIEVAL:
       - Query with partial pattern
       - Returns nearest stored pattern
       
    4. DIFFERENTIABLE:
       - Can be used in end-to-end training
       - Gradients flow through retrieval
    
    ═══════════════════════════════════════════════════════════════════════
    MATHEMATICAL FORMULATION
    ═══════════════════════════════════════════════════════════════════════
    
    Energy function:
        E(ξ) = -lse(β, X^T ξ) + 0.5 ξ^T ξ + const
        
    Where lse = log-sum-exp (smooth max)
    
    Update rule (one step convergence):
        ξ_new = X · softmax(β X^T ξ_old)
        
    This is exactly attention! Query=ξ, Keys=Values=X
    
    ═══════════════════════════════════════════════════════════════════════
    """
    
    def __init__(
        self,
        pattern_dim: int,
        memory_size: int = 10000,
        beta: float = 1.0,
        separate_keys_values: bool = False,
    ):
        super().__init__()
        
        self.pattern_dim = pattern_dim
        self.memory_size = memory_size
        self.beta = beta
        self.separate_kv = separate_keys_values
        
        # Memory storage
        self.register_buffer('patterns', torch.zeros(memory_size, pattern_dim))
        
        # If separate K/V, store both
        if separate_keys_values:
            self.register_buffer('values', torch.zeros(memory_size, pattern_dim))
        
        # Write pointer
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_stored', torch.tensor(0, dtype=torch.long))
        
        # Optional projections
        self.key_proj = None
        self.query_proj = None
    
    def store(
        self,
        pattern: torch.Tensor,
        value: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Store a pattern (and optionally a different value).
        
        Args:
            pattern: Pattern to store [pattern_dim] or [batch, pattern_dim]
            value: Optional separate value [pattern_dim] or [batch, pattern_dim]
            
        Returns:
            Index where pattern was stored
        """
        # Handle batched input
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
        
        batch_size = pattern.shape[0]
        
        for i in range(batch_size):
            idx = self.write_ptr.item()
            self.patterns[idx] = pattern[i].detach()
            
            if self.separate_kv and value is not None:
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                self.values[idx] = value[i].detach()
            
            # Advance pointer (circular buffer)
            self.write_ptr = (self.write_ptr + 1) % self.memory_size
            self.num_stored = torch.clamp(self.num_stored + 1, max=self.memory_size)
        
        return idx
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 1,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve pattern(s) most similar to query.
        
        Modern Hopfield update:
            ξ_new = X · softmax(β X^T query)
        
        Args:
            query: Query pattern [pattern_dim] or [batch, pattern_dim]
            top_k: Number of patterns to retrieve
            return_weights: If True, return attention weights
            
        Returns:
            retrieved: Retrieved pattern(s) [batch, pattern_dim]
            weights: Optional attention weights [batch, num_stored]
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Get valid patterns
        n = self.num_stored.item()
        if n == 0:
            return query.clone(), None
        
        patterns = self.patterns[:n]  # [n, pattern_dim]
        
        # Compute attention scores
        scores = self.beta * torch.matmul(query, patterns.T)  # [batch, n]
        
        # Top-k selection for efficiency
        if top_k < n:
            topk_scores, topk_idx = torch.topk(scores, top_k, dim=-1)
            weights = F.softmax(topk_scores, dim=-1)  # [batch, top_k]
            
            # Gather top-k patterns
            if self.separate_kv:
                values = self.values[:n]
                topk_patterns = values[topk_idx]  # [batch, top_k, pattern_dim]
            else:
                topk_patterns = patterns[topk_idx]  # [batch, top_k, pattern_dim]
            
            # Weighted combination
            retrieved = torch.bmm(weights.unsqueeze(1), topk_patterns).squeeze(1)
        else:
            weights = F.softmax(scores, dim=-1)  # [batch, n]
            
            if self.separate_kv:
                values = self.values[:n]
                retrieved = torch.matmul(weights, values)
            else:
                retrieved = torch.matmul(weights, patterns)
        
        if return_weights:
            return retrieved, weights
        return retrieved, None
    
    def energy(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of a pattern.
        
        Lower energy = more stable = better reconstructed
        
        E(ξ) = -lse(β, X^T ξ) + 0.5 ||ξ||²
        """
        n = self.num_stored.item()
        if n == 0:
            return pattern.pow(2).sum() * 0.5
        
        patterns = self.patterns[:n]
        
        # Similarity to stored patterns
        similarities = self.beta * torch.matmul(pattern, patterns.T)
        
        # Log-sum-exp (smooth max)
        lse = torch.logsumexp(similarities, dim=-1)
        
        # Energy
        energy = -lse + 0.5 * pattern.pow(2).sum(dim=-1)
        
        return energy
    
    def clear(self):
        """Clear all stored patterns."""
        self.patterns.zero_()
        if self.separate_kv:
            self.values.zero_()
        self.write_ptr.zero_()
        self.num_stored.zero_()
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        n = self.num_stored.item()
        
        stats = {
            'num_stored': n,
            'capacity': self.memory_size,
            'utilization': n / self.memory_size,
        }
        
        if n > 0:
            patterns = self.patterns[:n]
            # Pattern diversity
            norms = patterns.norm(dim=1)
            self_sim = torch.matmul(patterns, patterns.T)
            self_sim.fill_diagonal_(0)
            
            stats['mean_pattern_norm'] = norms.mean().item()
            stats['mean_similarity'] = self_sim.sum().item() / (n * (n - 1) + 1e-8)
        
        return stats


class WorkingMemoryBuffer(nn.Module):
    """
    Working Memory - Limited capacity, fast access buffer.
    
    Based on Miller's Law: 7±2 items in working memory.
    
    Implementation:
    - Fixed number of "slots"
    - Attention-based read/write
    - Decay over time (items forgotten)
    """
    
    def __init__(
        self,
        slot_dim: int,
        num_slots: int = 7,
        decay_rate: float = 0.01,
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.decay_rate = decay_rate
        
        # Working memory slots
        self.register_buffer('slots', torch.zeros(num_slots, slot_dim))
        
        # Slot activation/importance
        self.register_buffer('activation', torch.zeros(num_slots))
        
        # Read/write projections
        self.query_proj = nn.Linear(slot_dim, slot_dim)
        self.write_proj = nn.Linear(slot_dim, slot_dim)
    
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from working memory with attention.
        
        Args:
            query: Query vector [batch, slot_dim] or [slot_dim]
            
        Returns:
            content: Retrieved content [batch, slot_dim]
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Project query
        q = self.query_proj(query)
        
        # Attention over slots
        scores = torch.matmul(q, self.slots.T) / math.sqrt(self.slot_dim)
        
        # Weight by slot activation
        scores = scores + self.activation.unsqueeze(0).log().clamp(min=-10)
        
        weights = F.softmax(scores, dim=-1)
        
        # Read
        content = torch.matmul(weights, self.slots)
        
        return content
    
    def write(self, content: torch.Tensor, strength: float = 1.0) -> int:
        """
        Write to working memory.
        
        Writes to least active slot.
        
        Args:
            content: Content to store [slot_dim]
            strength: Write strength (higher = more permanent)
            
        Returns:
            slot_idx: Index of written slot
        """
        # Find least active slot
        slot_idx = self.activation.argmin().item()
        
        # Project content
        c = self.write_proj(content) if content.dim() > 1 else self.write_proj(content.unsqueeze(0)).squeeze(0)
        
        # Write
        self.slots[slot_idx] = c.detach()
        self.activation[slot_idx] = strength
        
        return slot_idx
    
    def step(self):
        """
        Time step: decay activations.
        
        Call this each forward step to simulate forgetting.
        """
        self.activation = self.activation * (1 - self.decay_rate)
    
    def clear(self):
        """Clear working memory."""
        self.slots.zero_()
        self.activation.zero_()


class MemoryConsolidation(nn.Module):
    """
    Memory Consolidation System (Sleep Replay).
    
    Transfers important episodic memories to semantic weights
    via repeated replay.
    """
    
    def __init__(
        self,
        episodic_memory: ModernHopfieldMemory,
        consolidation_rate: float = 0.001,
        theta_freq: float = 6.0,  # Theta rhythm for replay
    ):
        super().__init__()
        
        self.episodic = episodic_memory
        self.rate = consolidation_rate
        self.theta_freq = theta_freq
        
        # Track which patterns have been consolidated
        self.register_buffer(
            'consolidation_count',
            torch.zeros(episodic_memory.memory_size, dtype=torch.long)
        )
    
    def replay_batch(
        self,
        batch_size: int = 32,
        semantic_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Replay a batch of episodic memories.
        
        If semantic_weights provided, updates them Hebbianly.
        
        Args:
            batch_size: Number of memories to replay
            semantic_weights: Optional weight matrix to update [out, in]
            
        Returns:
            replayed: The replayed patterns [batch, pattern_dim]
            info: Statistics about replay
        """
        n = self.episodic.num_stored.item()
        if n == 0:
            return None, {'error': 'No memories to replay'}
        
        # Sample memories (weighted by importance = 1/consolidation_count)
        counts = self.consolidation_count[:n].float() + 1
        probs = 1.0 / counts
        probs = probs / probs.sum()
        
        indices = torch.multinomial(probs, min(batch_size, n), replacement=False)
        replayed = self.episodic.patterns[indices]
        
        # Update consolidation counts
        self.consolidation_count[indices] += 1
        
        # Theta modulation
        t = torch.linspace(0, 2 * math.pi, batch_size, device=replayed.device)
        theta_mod = 0.5 + 0.5 * torch.sin(self.theta_freq * t)
        
        # Apply theta modulation
        replayed = replayed * theta_mod.unsqueeze(1)
        
        # Update semantic weights if provided
        if semantic_weights is not None:
            # Hebbian update from replay
            # Δw ~ replay @ replay^T
            with torch.no_grad():
                hebbian = torch.matmul(replayed.T, replayed) / batch_size
                
                # Scale by (1 - |w|) to prevent saturation
                space = 1 - semantic_weights.abs().clamp(max=0.99)
                delta = self.rate * hebbian * space
                
                semantic_weights.add_(delta)
        
        info = {
            'num_replayed': len(indices),
            'mean_consolidation': self.consolidation_count[:n].float().mean().item(),
            'theta_mean': theta_mod.mean().item(),
        }
        
        return replayed, info
    
    def consolidate_cycle(
        self,
        semantic_weights: torch.Tensor,
        n_iterations: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """
        Run a full consolidation cycle (simulated sleep).
        
        Args:
            semantic_weights: Weight matrix to update
            n_iterations: Number of replay iterations
            batch_size: Patterns per replay
            
        Returns:
            stats: Consolidation statistics
        """
        total_weight_change = 0.0
        initial_weights = semantic_weights.clone()
        
        for i in range(n_iterations):
            replayed, info = self.replay_batch(batch_size, semantic_weights)
            if replayed is None:
                break
        
        weight_change = (semantic_weights - initial_weights).abs().mean().item()
        
        return {
            'iterations': n_iterations,
            'total_weight_change': weight_change,
            'mean_consolidation': self.consolidation_count[:self.episodic.num_stored].float().mean().item(),
        }


class TripleMemorySystem(nn.Module):
    """
    Complete triple memory system:
    1. Working Memory (fast, limited)
    2. Episodic Memory (one-shot, large)
    3. Semantic Memory (slow, permanent)
    """
    
    def __init__(
        self,
        pattern_dim: int,
        working_slots: int = 7,
        episodic_capacity: int = 10000,
        semantic_dim: int = 256,
    ):
        super().__init__()
        
        self.pattern_dim = pattern_dim
        
        # Working memory
        self.working = WorkingMemoryBuffer(
            slot_dim=pattern_dim,
            num_slots=working_slots,
        )
        
        # Episodic memory
        self.episodic = ModernHopfieldMemory(
            pattern_dim=pattern_dim,
            memory_size=episodic_capacity,
        )
        
        # Semantic memory (as weight matrix)
        self.semantic_weights = nn.Parameter(
            torch.randn(semantic_dim, pattern_dim) * 0.01
        )
        
        # Consolidation system
        self.consolidation = MemoryConsolidation(self.episodic)
        
        # Gating network: decides where to store
        self.gating = nn.Sequential(
            nn.Linear(pattern_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [working, episodic, semantic]
        )
    
    def process(
        self,
        input_pattern: torch.Tensor,
        store: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through triple memory.
        
        Args:
            input_pattern: Input [pattern_dim]
            store: Whether to store the pattern
            
        Returns:
            output: Processed pattern (enriched by memory)
            info: Processing information
        """
        # Read from working memory
        working_content = self.working.read(input_pattern)
        
        # Read from episodic
        episodic_content, _ = self.episodic.retrieve(input_pattern)
        
        # Semantic transform
        semantic_content = F.linear(input_pattern.unsqueeze(0) if input_pattern.dim() == 1 else input_pattern, 
                                    self.semantic_weights)
        
        # Combine
        combined = 0.5 * input_pattern + 0.2 * working_content.squeeze() + 0.2 * episodic_content.squeeze() + 0.1 * semantic_content.squeeze()
        
        # Store decision
        if store:
            gate_logits = self.gating(input_pattern)
            gate_probs = F.softmax(gate_logits, dim=-1)
            
            # Probabilistic storage
            if torch.rand(1).item() < gate_probs[0]:
                self.working.write(input_pattern)
            if torch.rand(1).item() < gate_probs[1]:
                self.episodic.store(input_pattern)
        
        # Decay working memory
        self.working.step()
        
        info = {
            'working_activation': self.working.activation.mean().item(),
            'episodic_stored': self.episodic.num_stored.item(),
        }
        
        return combined, info
    
    def sleep(self, n_iterations: int = 100):
        """Run consolidation (sleep) cycle."""
        return self.consolidation.consolidate_cycle(
            self.semantic_weights.data,
            n_iterations=n_iterations,
        )
