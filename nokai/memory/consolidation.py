"""
Consolidation System - Sleep and Memory Transfer

Biological Parallel:
    During sleep, the brain consolidates memories:
    
    1. Hippocampal Replay: Episodic memories are "replayed"
    2. Neocortical Integration: Slow integration into semantic memory
    3. Synaptic Homeostasis: Weak synapses pruned, strong preserved
    4. Memory Transformation: Episodes → Abstractions
    
Implementation:
    A background process that:
    1. Replays recent experiences from episodic buffer
    2. Consolidates to semantic memory
    3. Applies synaptic homeostasis (normalization)
    4. Optimizes plasticity
    
Efficiency:
    - Runs asynchronously during "idle" time
    - Batch processing for efficiency
    - Minimal impact on real-time processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
import threading
import time
from queue import Queue
from dataclasses import dataclass

if TYPE_CHECKING:
    from nokai.memory.semantic import SemanticMemory
    from nokai.hippocampus import HippocampalMemory


@dataclass
class ConsolidationEvent:
    """A memory to be consolidated."""
    content: torch.Tensor
    importance: float
    source: str  # "episodic" or "working"
    timestamp: float


class ConsolidationSystem(nn.Module):
    """
    Sleep Consolidation System
    
    Biological Parallel:
        During sleep (especially slow-wave and REM), the brain:
        
        1. Hippocampal-Neocortical Transfer
           - Hippocampus replays recent episodes
           - Neocortex slowly integrates patterns
           - Specific details lost, abstractions preserved
           
        2. Synaptic Homeostasis (SHY Hypothesis)
           - Waking strengthens synapses (LTP)
           - Sleep globally downscales (renormalization)
           - Only important connections remain strong
           
        3. Memory Transformation
           - Episodic → Semantic (facts from experiences)
           - Procedural optimization (skill refinement)
           - Creative integration (insight)
           
    Implementation:
        - Background consolidation thread
        - Hippocampus → Semantic memory transfer
        - Synaptic weight normalization
        - Working memory clearance
        
    Efficiency:
        - Runs during "idle" periods
        - Batch processing
        - Minimal real-time overhead
    """
    
    def __init__(
        self,
        embedding_dim: int,
        consolidation_rate: float = 0.1,
        homeostasis_rate: float = 0.01,
        replay_batch_size: int = 32,
        run_async: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.consolidation_rate = consolidation_rate
        self.homeostasis_rate = homeostasis_rate
        self.replay_batch_size = replay_batch_size
        
        # ============================================
        # MEMORY ABSTRACTOR
        # ============================================
        # Transforms episodic memories to semantic representations
        self.abstractor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # ============================================
        # IMPORTANCE SCORER
        # ============================================
        # Determines which memories to consolidate
        self.importance_scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # ============================================
        # CONSOLIDATION QUEUE
        # ============================================
        self.consolidation_queue: Queue = Queue(maxsize=10000)
        
        # ============================================
        # STATISTICS
        # ============================================
        self.register_buffer('total_consolidated', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_pruned', torch.tensor(0, dtype=torch.long))
        self.register_buffer('consolidation_sessions', torch.tensor(0, dtype=torch.long))
        
        # Background thread (if async)
        self.run_async = run_async
        self._running = False
        self._thread = None
    
    def start_async(self):
        """Start background consolidation thread."""
        if self.run_async and not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._consolidation_loop, daemon=True)
            self._thread.start()
    
    def stop_async(self):
        """Stop background consolidation thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _consolidation_loop(self):
        """Background consolidation loop."""
        while self._running:
            if not self.consolidation_queue.empty():
                self.consolidate_step()
            time.sleep(0.1)  # Avoid busy-waiting
    
    def queue_for_consolidation(
        self,
        content: torch.Tensor,
        importance: Optional[float] = None,
        source: str = "episodic",
    ):
        """
        Queue a memory for consolidation.
        
        Args:
            content: Memory content [dim] or [batch, dim]
            importance: Pre-computed importance score
            source: Source of memory ("episodic" or "working")
        """
        if content.dim() == 1:
            content = content.unsqueeze(0)
        
        for i in range(content.shape[0]):
            c = content[i].detach().clone()
            
            if importance is None:
                with torch.no_grad():
                    imp = self.importance_scorer(c).item()
            else:
                imp = importance
            
            event = ConsolidationEvent(
                content=c,
                importance=imp,
                source=source,
                timestamp=time.time(),
            )
            
            try:
                self.consolidation_queue.put_nowait(event)
            except:
                pass  # Queue full, skip
    
    def consolidate_step(
        self,
        semantic_memory: Optional['SemanticMemory'] = None,
        hippocampus: Optional['HippocampalMemory'] = None,
    ) -> Dict:
        """
        Perform one consolidation step.
        
        This simulates a "sleep" micro-session where:
        1. Recent memories are replayed
        2. Important ones are consolidated to semantic memory
        3. Synaptic homeostasis is applied
        
        Args:
            semantic_memory: Target semantic memory module
            hippocampus: Source episodic memory
            
        Returns:
            metadata: Consolidation statistics
        """
        consolidated = 0
        pruned = 0
        
        # ============================================
        # REPLAY AND CONSOLIDATE
        # ============================================
        batch = []
        while len(batch) < self.replay_batch_size and not self.consolidation_queue.empty():
            try:
                event = self.consolidation_queue.get_nowait()
                batch.append(event)
            except:
                break
        
        if len(batch) > 0:
            # Stack contents
            contents = torch.stack([e.content for e in batch])
            importances = torch.tensor([e.importance for e in batch])
            
            # Filter by importance
            threshold = 0.3
            important_mask = importances > threshold
            
            if important_mask.any():
                important_contents = contents[important_mask]
                
                # Abstract the memories
                with torch.no_grad():
                    abstracted = self.abstractor(important_contents)
                
                # Store in semantic memory
                if semantic_memory is not None:
                    for i in range(abstracted.shape[0]):
                        semantic_memory.store(abstracted[i], immediate=True)
                        consolidated += 1
            
            pruned = (~important_mask).sum().item()
        
        # ============================================
        # SYNAPTIC HOMEOSTASIS
        # ============================================
        # Would apply to all modules with synaptic weights
        # This is a placeholder for the actual implementation
        
        # Update statistics
        self.total_consolidated += consolidated
        self.total_pruned += pruned
        self.consolidation_sessions += 1
        
        metadata = {
            'consolidated': consolidated,
            'pruned': pruned,
            'queue_size': self.consolidation_queue.qsize(),
            'total_consolidated': self.total_consolidated.item(),
            'total_pruned': self.total_pruned.item(),
        }
        
        return metadata
    
    def apply_synaptic_homeostasis(
        self,
        modules: List[nn.Module],
    ) -> Dict:
        """
        Apply synaptic homeostasis to all modules.
        
        Biological Parallel:
            During sleep, synapses are globally downscaled (SHY hypothesis).
            This prevents runaway weight growth while preserving relative
            differences (important connections stay stronger).
            
        Args:
            modules: List of modules with 'synaptic_weights' buffers
            
        Returns:
            metadata: Homeostasis statistics
        """
        total_scaled = 0
        
        for module in modules:
            if hasattr(module, 'synaptic_weights'):
                with torch.no_grad():
                    weights = module.synaptic_weights
                    
                    # Global downscaling
                    mean_weight = weights.mean()
                    if mean_weight > 1.0:
                        scale_factor = 1.0 / mean_weight
                        weights.mul_(scale_factor ** self.homeostasis_rate)
                        
                        # Ensure minimum weight
                        weights.clamp_(min=0.5)
                        
                        total_scaled += weights.numel()
        
        return {
            'total_scaled': total_scaled,
        }
    
    def full_consolidation_session(
        self,
        semantic_memory: Optional['SemanticMemory'] = None,
        hippocampus: Optional['HippocampalMemory'] = None,
        modules_for_homeostasis: Optional[List[nn.Module]] = None,
        max_steps: int = 100,
    ) -> Dict:
        """
        Run a full consolidation session (simulate sleep).
        
        Args:
            semantic_memory: Target for consolidated memories
            hippocampus: Source of episodic memories
            modules_for_homeostasis: Modules to apply homeostasis
            max_steps: Maximum consolidation steps
            
        Returns:
            metadata: Session statistics
        """
        total_consolidated = 0
        total_pruned = 0
        
        for step in range(max_steps):
            if self.consolidation_queue.empty():
                break
            
            meta = self.consolidate_step(semantic_memory, hippocampus)
            total_consolidated += meta['consolidated']
            total_pruned += meta['pruned']
        
        # Apply homeostasis
        homeostasis_meta = {}
        if modules_for_homeostasis:
            homeostasis_meta = self.apply_synaptic_homeostasis(modules_for_homeostasis)
        
        # Clear semantic memory consolidation buffer
        if semantic_memory is not None:
            semantic_memory.consolidate(batch_size=1000)
        
        return {
            'total_consolidated': total_consolidated,
            'total_pruned': total_pruned,
            'steps': step + 1,
            **homeostasis_meta,
        }
    
    def energy_check(self) -> bool:
        """Check if consolidation is needed."""
        return not self.consolidation_queue.empty()


class SynapticHomeostasis(nn.Module):
    """
    Synaptic Homeostasis Helper
    
    Applies homeostatic scaling to maintain network stability.
    Used during consolidation to prevent runaway weight growth.
    """
    
    def __init__(self, target_mean: float = 1.0, rate: float = 0.01):
        super().__init__()
        self.target_mean = target_mean
        self.rate = rate
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply homeostatic scaling."""
        current_mean = weights.mean()
        
        if current_mean > self.target_mean:
            scale = self.target_mean / current_mean
            scaled_weights = weights * (scale ** self.rate)
        else:
            scaled_weights = weights
        
        return scaled_weights.clamp(min=0.5, max=2.0)
