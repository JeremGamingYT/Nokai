"""
Prefrontal Cortex - Working Memory System

Biological Parallel:
    The prefrontal cortex (PFC) maintains information "online" in
    working memory. Unlike long-term memory, working memory:
    
    1. Has limited capacity (7±2 items classically)
    2. Requires active maintenance (decays quickly)
    3. Is gated - decides what enters/exits
    4. Is modulated by dopamine (D1 receptors)
    
Implementation:
    We implement a gated working memory buffer with:
    1. Fixed capacity (configurable)
    2. Importance-based slot allocation
    3. Active recurrency to prevent decay
    4. Gating mechanism for updates
    
Efficiency:
    - O(C) where C = capacity (typically small: 8-16)
    - Constant memory footprint
    - Sparse updates only when needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class WorkingMemorySlot(nn.Module):
    """
    A single slot in working memory.
    
    Biological Parallel:
        A "slot" represents a coherent chunk of information
        being actively maintained by PFC neural activity.
        
    Properties:
        - content: The stored information
        - strength: How strongly maintained (decay rate)
        - age: How long since last update
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Slot content buffer
        self.register_buffer('content', torch.zeros(dim))
        
        # Slot metadata
        self.register_buffer('strength', torch.tensor(0.0))
        self.register_buffer('age', torch.tensor(0.0))
        self.register_buffer('is_occupied', torch.tensor(False))
    
    def write(self, content: torch.Tensor, strength: float = 1.0):
        """Write content to slot."""
        self.content = content.squeeze()
        self.strength = torch.tensor(strength)
        self.age = torch.tensor(0.0)
        self.is_occupied = torch.tensor(True)
    
    def read(self) -> torch.Tensor:
        """Read content from slot."""
        return self.content
    
    def decay(self, rate: float = 0.95):
        """Apply decay to slot strength."""
        if self.is_occupied:
            self.strength = self.strength * rate
            self.age = self.age + 1
            
            # Clear if strength too low
            if self.strength < 0.1:
                self.clear()
    
    def clear(self):
        """Clear the slot."""
        self.content.zero_()
        self.strength = torch.tensor(0.0)
        self.age = torch.tensor(0.0)
        self.is_occupied = torch.tensor(False)
    
    def refresh(self):
        """Refresh the slot (re-activate)."""
        if self.is_occupied:
            self.strength = torch.tensor(1.0)
            self.age = torch.tensor(0.0)


class PrefrontalWorkingMemory(nn.Module):
    """
    Prefrontal Cortex Working Memory
    
    Biological Parallel:
        The PFC maintains task-relevant information through sustained
        neural firing. This is working memory - the "scratch pad" of
        cognition.
        
        Key mechanisms:
        1. Input gating: What enters working memory
        2. Output gating: What influences processing
        3. Forget gating: What gets removed
        4. Recurrent maintenance: Active refreshing
        
    Implementation:
        A differentiable memory buffer with:
        - Fixed capacity (slot-based)
        - LSTM-like gating for updates
        - Attention-based retrieval
        - Decay without maintenance
        
    Efficiency:
        - O(C) complexity (C = capacity)
        - Constant memory regardless of sequence length
        - Sparse updates (only when gated)
    """
    
    def __init__(
        self,
        dim: int,
        capacity: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        
        # ============================================
        # MEMORY BUFFER
        # ============================================
        self.register_buffer('memory', torch.zeros(capacity, dim))
        self.register_buffer('memory_strengths', torch.zeros(capacity))
        self.register_buffer('memory_ages', torch.zeros(capacity))
        
        # ============================================
        # GATING NETWORKS
        # ============================================
        # Input gate: Should this information enter WM?
        self.input_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        
        # Forget gate: Should we forget old information?
        self.forget_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        
        # Output gate: How much does WM influence output?
        self.output_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        
        # ============================================
        # CONTENT PROCESSING
        # ============================================
        # Transform input for storage
        self.input_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
        )
        
        # ============================================
        # RETRIEVAL ATTENTION
        # ============================================
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # ============================================
        # SLOT IMPORTANCE CALCULATOR
        # ============================================
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        
        # ============================================
        # SYNAPTIC WEIGHTS (Plasticity)
        # ============================================
        # Track which slots are frequently useful
        self.register_buffer('slot_utilization', torch.zeros(capacity))
        self.register_buffer('synaptic_weights', torch.ones(capacity))
        
        # Decay rate (can be modulated by dopamine)
        self.base_decay_rate = 0.95
    
    def forward(
        self,
        x: torch.Tensor,
        dopamine_level: float = 0.5,
        store: bool = True,
        retrieve: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through working memory.
        
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, dim]
            dopamine_level: Current dopamine (modulates gating)
            store: Whether to store new information
            retrieve: Whether to retrieve from memory
            
        Returns:
            output: WM-augmented output
            metadata: Processing details
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq dimension
        
        batch_size, seq_len, dim = x.shape
        
        # Get current memory state (expand for batch)
        current_memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        current_strengths = self.memory_strengths.unsqueeze(0).expand(batch_size, -1)
        
        outputs = []
        store_events = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, dim]
            
            # ============================================
            # RETRIEVAL (if enabled)
            # ============================================
            retrieved = torch.zeros_like(x_t)
            if retrieve and (current_strengths.max() > 0.1):
                # Attention-based retrieval
                query = self.query_proj(x_t.unsqueeze(1))
                
                # Mask empty slots
                mask = current_strengths < 0.1  # [batch, capacity]
                
                retrieved, attn_weights = self.attention(
                    query,
                    current_memory,
                    current_memory,
                    key_padding_mask=mask,
                )
                retrieved = retrieved.squeeze(1)
            
            # ============================================
            # OUTPUT GATING
            # ============================================
            combined_for_gate = torch.cat([x_t, retrieved], dim=-1)
            output_g = self.output_gate(combined_for_gate)
            
            out_t = x_t + output_g * retrieved
            outputs.append(out_t)
            
            # ============================================
            # INPUT GATING (decide what to store)
            # ============================================
            if store:
                # Compute input importance
                importance = self.importance_net(x_t).squeeze(-1)
                
                # Gate: Should we store this?
                input_g = self.input_gate(combined_for_gate)
                store_value = self.input_transform(x_t * input_g)
                
                # Find best slot (least important currently stored)
                slot_scores = current_strengths / (self.slot_utilization + 1)
                worst_slot = slot_scores.argmin(dim=-1)  # [batch]
                
                # Update memory (batch processing)
                for b in range(batch_size):
                    slot = worst_slot[b].item()
                    if importance[b] > current_strengths[b, slot]:
                        current_memory[b, slot] = store_value[b]
                        current_strengths[b, slot] = importance[b].clamp(0, 1)
                        store_events.append((b, slot))
                        
                        # Update utilization for plasticity
                        self.slot_utilization[slot] += 1
        
        # ============================================
        # DECAY AND MAINTENANCE
        # ============================================
        # Dopamine modulates decay rate (higher DA = slower decay)
        decay_rate = self.base_decay_rate + 0.04 * dopamine_level
        self.memory_strengths = self.memory_strengths * decay_rate
        self.memory_ages = self.memory_ages + 1
        
        # Clear very weak memories
        weak_mask = self.memory_strengths < 0.1
        self.memory[weak_mask] = 0
        self.memory_strengths[weak_mask] = 0
        self.memory_ages[weak_mask] = 0
        
        # Update synaptic weights (LTP for useful slots)
        if self.training and len(store_events) > 0:
            for _, slot in store_events:
                self.synaptic_weights[slot] = min(
                    self.synaptic_weights[slot] * 1.01,
                    2.0
                )
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        if seq_len == 1:
            output = output.squeeze(1)
        
        # Update stored memory with one representative sample
        self.memory = current_memory[0].detach()
        self.memory_strengths = current_strengths[0].detach()
        
        metadata = {
            'occupied_slots': (self.memory_strengths > 0.1).sum().item(),
            'mean_strength': self.memory_strengths[self.memory_strengths > 0.1].mean().item()
                if (self.memory_strengths > 0.1).any() else 0.0,
            'oldest_age': self.memory_ages.max().item(),
            'store_events': len(store_events),
            'decay_rate': decay_rate,
            'synaptic_weights_mean': self.synaptic_weights.mean().item(),
        }
        
        return output, metadata
    
    def refresh_all(self):
        """Refresh all slots (reset decay)."""
        self.memory_strengths.clamp_(min=0.0)
        active = self.memory_strengths > 0.1
        self.memory_strengths[active] = 1.0
        self.memory_ages[active] = 0
    
    def clear(self):
        """Clear all working memory."""
        self.memory.zero_()
        self.memory_strengths.zero_()
        self.memory_ages.zero_()
    
    def energy_check(self) -> bool:
        """Check if WM processing is needed."""
        # If memory is empty, minimal processing needed
        return (self.memory_strengths > 0.1).any().item()
    
    def get_contents(self) -> List[torch.Tensor]:
        """Get current WM contents (for debugging)."""
        active = self.memory_strengths > 0.1
        return self.memory[active]


class ContextBuffer(nn.Module):
    """
    Simplified context buffer for short-term context maintenance.
    
    Less biologically inspired but more efficient for simple cases.
    Used when full WM processing isn't needed.
    """
    
    def __init__(self, dim: int, max_length: int = 64):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        
        self.register_buffer('buffer', torch.zeros(max_length, dim))
        self.register_buffer('position', torch.tensor(0, dtype=torch.long))
        self.register_buffer('length', torch.tensor(0, dtype=torch.long))
    
    def push(self, x: torch.Tensor):
        """Add to buffer (FIFO)."""
        if x.dim() > 1:
            x = x.mean(dim=tuple(range(x.dim() - 1)))
        
        self.buffer[self.position] = x
        self.position = (self.position + 1) % self.max_length
        self.length = min(self.length + 1, self.max_length)
    
    def get(self) -> torch.Tensor:
        """Get buffer contents in order."""
        if self.length == 0:
            return torch.zeros(1, self.dim, device=self.buffer.device)
        
        return self.buffer[:self.length]
    
    def clear(self):
        self.buffer.zero_()
        self.position.zero_()
        self.length.zero_()
