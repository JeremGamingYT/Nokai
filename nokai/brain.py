"""
Neuromorphic Brain - Complete Bio-Inspired AGI Architecture

This module integrates all neuromorphic components into a unified
cognitive architecture inspired by the human brain.

Architecture Overview:
    ┌──────────────────────────────────────────────────────────────┐
    │                     NŌKAI NEUROMORPHIC BRAIN                  │
    ├──────────────────────────────────────────────────────────────┤
    │  INPUT → [THALAMUS] → Filter/Route                           │
    │            ↓                                                  │
    │  [CORTEX] ←→ [WORKING MEMORY] ←→ [HIPPOCAMPUS]              │
    │      ↓              ↓                   ↓                    │
    │  [SEMANTIC] ←── [CONSOLIDATION] ←── [EPISODIC]              │
    │            ↓                                                  │
    │  [dACC] → Uncertainty → [ATTENTION CONTROLLER]               │
    │            ↓                                                  │
    │  [STRIATUM] ←── [DOPAMINE/VTA] → Action Selection            │
    │            ↓                                                  │
    │  OUTPUT ← Decision/Response                                   │
    └──────────────────────────────────────────────────────────────┘
    
Key Features:
    1. Plasticité Synaptique: All modules have synaptic_weights
    2. Sparsité/Efficacité: energy_check() before activation
    3. Dopamine: Modulates learning, attention, decisions
    4. Oscillations: Coordinate processing across modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import time

from nokai.config import NokaiConfig
from nokai.cortex import Cortex
from nokai.hippocampus import HippocampalMemory
from nokai.oscillations import OscillatorNetwork
from nokai.thalamus import ThalamusGateway
from nokai.prefrontal import PrefrontalWorkingMemory
from nokai.limbic import DopamineCircuit, StriatumSelector, MetacognitiveMonitor
from nokai.memory import SemanticMemory, ConsolidationSystem
from nokai.attention import AttentionController, ResourceAllocation


@dataclass
class BrainState:
    """
    Complete state of the neuromorphic brain.
    
    Biological Mapping:
        - dopamine_level: Motivation/reward drive
        - arousal: Alertness level
        - confidence: Certainty in current processing
        - cognitive_load: Current processing demands
        - oscillation_phase: Neural synchronization state
    """
    dopamine_level: float = 0.5
    arousal: float = 0.5
    confidence: float = 0.5
    cognitive_load: float = 0.5
    oscillation_phase: float = 0.0
    active_modules: List[str] = None
    
    def __post_init__(self):
        if self.active_modules is None:
            self.active_modules = []


class NeuromorphicBrain(nn.Module):
    """
    Complete Neuromorphic Brain Architecture
    
    This is the main integration point for all bio-inspired modules.
    It orchestrates the flow of information through the cognitive
    hierarchy, implementing the full AGI architecture.
    
    Key Mechanisms:
    
    1. SENSORY PROCESSING (Thalamus)
       - Filters irrelevant inputs
       - Routes to appropriate cortical areas
       - Implements attentional gating
       
    2. CORTICAL PROCESSING (Cortex)
       - Hierarchical feature extraction
       - Predictive coding
       - Sparse distributed representations
       
    3. MEMORY SYSTEMS
       - Working Memory (Prefrontal): Active maintenance
       - Episodic Memory (Hippocampus): Event storage
       - Semantic Memory (Neocortex): Knowledge base
       - Consolidation: Memory transfer during "sleep"
       
    4. EXECUTIVE CONTROL
       - Metacognition (dACC): Uncertainty monitoring
       - Attention Controller: Resource allocation
       - Decision Making (Striatum): Action selection
       
    5. MOTIVATION
       - Dopamine Circuit: Reward prediction, learning modulation
       
    6. SYNCHRONIZATION
       - Neural Oscillations: Coordinate processing
       
    Efficiency:
        - Sparse activation (energy_check for each module)
        - Dynamic resource allocation
        - O(N log N) attention instead of O(N²)
        - Memory-mapped storage for scale
    """
    
    def __init__(self, config: NokaiConfig):
        super().__init__()
        self.config = config
        
        # ============================================
        # EMBEDDING LAYER
        # ============================================
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.embed_norm = nn.LayerNorm(config.embedding_dim)
        self.embed_dropout = nn.Dropout(config.column_config.dropout)
        
        # ============================================
        # THALAMUS - Sensory Gateway
        # ============================================
        self.thalamus = ThalamusGateway(
            input_dim=config.embedding_dim,
            num_clusters=config.thalamus.num_clusters,
            sparsity_target=config.thalamus.sparsity_target,
            oscillation_coupling=config.thalamus.oscillation_coupling,
        )
        
        # ============================================
        # CORTEX - Hierarchical Processing
        # ============================================
        self.cortex = Cortex(config)
        
        # ============================================
        # WORKING MEMORY - Prefrontal Cortex
        # ============================================
        self.working_memory = PrefrontalWorkingMemory(
            dim=config.embedding_dim,
            capacity=8,
            num_heads=4,
        )
        
        # ============================================
        # EPISODIC MEMORY - Hippocampus
        # ============================================
        if config.hippocampus.enabled:
            self.hippocampus = HippocampalMemory(
                embedding_dim=config.embedding_dim,
                memory_size=config.hippocampus.memory_size,
                num_heads=config.hippocampus.num_heads_ca3,
                retrieval_top_k=config.hippocampus.retrieval_top_k,
            )
        else:
            self.hippocampus = None
        
        # ============================================
        # SEMANTIC MEMORY - Neocortex
        # ============================================
        self.semantic_memory = SemanticMemory(
            embedding_dim=config.embedding_dim,
            max_concepts=100_000,  # Can scale with mmap
            update_rate=0.001,
        )
        
        # ============================================
        # CONSOLIDATION - Sleep System
        # ============================================
        self.consolidation = ConsolidationSystem(
            embedding_dim=config.embedding_dim,
            consolidation_rate=0.1,
        )
        
        # ============================================
        # DOPAMINE - Reward Circuit
        # ============================================
        self.dopamine_circuit = DopamineCircuit(
            state_dim=config.embedding_dim,
            hidden_dim=256,
        )
        
        # ============================================
        # STRIATUM - Decision Making
        # ============================================
        self.striatum = StriatumSelector(
            state_dim=config.embedding_dim,
            action_dim=config.embedding_dim,
            num_action_candidates=16,
        )
        
        # ============================================
        # dACC - Metacognition
        # ============================================
        self.dacc = MetacognitiveMonitor(
            state_dim=config.embedding_dim,
            hidden_dim=256,
        )
        
        # ============================================
        # ATTENTION CONTROLLER
        # ============================================
        self.attention_controller = AttentionController(
            state_dim=config.embedding_dim,
            num_modules=7,
        )
        
        # ============================================
        # NEURAL OSCILLATIONS
        # ============================================
        if config.oscillations.enabled:
            self.oscillations = OscillatorNetwork(
                num_oscillators=min(config.num_columns, 256),
                theta_freq=config.oscillations.theta_freq,
                gamma_freq=config.oscillations.gamma_freq,
                coupling_strength=config.oscillations.coupling_strength,
            )
        else:
            self.oscillations = None
        
        # ============================================
        # OUTPUT HEAD
        # ============================================
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Weight tying
        if config.shared_embeddings:
            self.output_projection.weight = self.embedding.weight
        
        # ============================================
        # BRAIN STATE
        # ============================================
        self.register_buffer('current_dopamine', torch.tensor(0.5))
        self.register_buffer('current_arousal', torch.tensor(0.5))
        
        # ============================================
        # GRADIENT CHECKPOINTING
        # ============================================
        self.gradient_checkpointing = config.memory_optimization.gradient_checkpointing
        
        # Print architecture summary
        self._print_architecture()
    
    def _print_architecture(self):
        """Print architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("NŌKAI NEUROMORPHIC BRAIN")
        print("=" * 60)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Modules:")
        print(f"    ├── Thalamus (Gateway)")
        print(f"    ├── Cortex (Processing)")
        print(f"    ├── Working Memory (PFC)")
        print(f"    ├── Episodic Memory (Hippocampus)")
        print(f"    ├── Semantic Memory (Neocortex)")
        print(f"    ├── Dopamine Circuit (VTA)")
        print(f"    ├── Striatum (Decisions)")
        print(f"    ├── dACC (Metacognition)")
        print(f"    ├── Attention Controller")
        print(f"    └── Oscillations")
        print("=" * 60)
    
    def embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens with position encoding."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        tok_emb = self.embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(positions)
        
        # Combine
        x = tok_emb + pos_emb
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        return x
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        store_memory: bool = False,
        return_brain_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Full forward pass through the neuromorphic brain.
        
        Architecture Flow:
            1. Embed input tokens
            2. Route through thalamus (filter)
            3. Process through cortex
            4. Integrate with working memory
            5. Retrieve from episodic/semantic memory
            6. Metacognitive assessment
            7. Update dopamine based on reward
            8. Generate output
            
        Args:
            input_ids: Token IDs [batch, seq]
            labels: Target labels for loss computation
            reward: Optional reward signal for dopamine update
            store_memory: Whether to store in memory systems
            return_brain_state: Return full brain state
            
        Returns:
            Dict with logits, loss, and optional brain state/metadata
        """
        # Embed input
        x = self.embed_input(input_ids)
        batch_size, seq_len, dim = x.shape
        
        # ============================================
        # UPDATE OSCILLATIONS
        # ============================================
        oscillation_phase = None
        if self.oscillations is not None:
            self.oscillations.step()
            oscillation_phase = self.oscillations.theta_phases.mean().item()
        
        # ============================================
        # ATTENTION CONTROL - Decide resource allocation
        # ============================================
        dopamine_level = self.current_dopamine.item()
        
        if self.attention_controller.energy_check(x):
            allocation, alloc_meta = self.attention_controller(
                x.mean(dim=1),  # Aggregate state
                dopamine_level=dopamine_level,
            )
        else:
            allocation = ResourceAllocation()  # Defaults
            alloc_meta = {}
        
        # ============================================
        # THALAMUS - Filter and route
        # ============================================
        if self.thalamus.energy_check(x):
            filtered_x, thalamus_meta = self.thalamus(
                x, 
                oscillation_phase=oscillation_phase,
            )
            # Pad back to original length if needed
            if filtered_x.shape[1] < seq_len:
                padding = torch.zeros(
                    batch_size, 
                    seq_len - filtered_x.shape[1], 
                    dim,
                    device=x.device
                )
                x = torch.cat([filtered_x, padding], dim=1)
            else:
                x = filtered_x
        else:
            thalamus_meta = {}
        
        # ============================================
        # CORTEX - Hierarchical processing
        # ============================================
        if self.gradient_checkpointing and self.training:
            cortex_out, cortex_meta = torch.utils.checkpoint.checkpoint(
                self.cortex, x, oscillation_phase,
                use_reentrant=False
            )
        else:
            cortex_out, cortex_meta = self.cortex(x, oscillation_phase=oscillation_phase)
        
        x = cortex_out
        
        # ============================================
        # WORKING MEMORY - Maintain context
        # ============================================
        if self.working_memory.energy_check() or allocation.prefrontal > 0.3:
            x, wm_meta = self.working_memory(
                x, 
                dopamine_level=dopamine_level,
                store=store_memory,
            )
        else:
            wm_meta = {}
        
        # ============================================
        # EPISODIC MEMORY - Retrieve relevant experiences
        # ============================================
        hippocampus_meta = {}
        if self.hippocampus is not None and allocation.hippocampus > 0.2:
            x, hippocampus_meta = self.hippocampus(x, store=store_memory)
        
        # ============================================
        # SEMANTIC MEMORY - Retrieve knowledge
        # ============================================
        if self.semantic_memory.energy_check() and allocation.semantic_memory > 0.1:
            x, semantic_meta = self.semantic_memory(
                x, 
                store=store_memory,
                retrieve=True,
            )
        else:
            semantic_meta = {}
        
        # ============================================
        # METACOGNITION - Assess uncertainty
        # ============================================
        if self.dacc.energy_check(x):
            assessment, dacc_meta = self.dacc(x.mean(dim=1))
        else:
            assessment = None
            dacc_meta = {}
        
        # ============================================
        # DOPAMINE UPDATE - Reward processing
        # ============================================
        if reward is not None:
            da_state, da_meta = self.dopamine_circuit(
                x.mean(dim=1),
                reward=reward,
            )
            self.current_dopamine.fill_(da_state.level)
        else:
            da_meta = {'dopamine_level': dopamine_level}
        
        # ============================================
        # QUEUE FOR CONSOLIDATION - Prepare for "sleep"
        # ============================================
        if store_memory and self.training:
            importance = dacc_meta.get('confidence', 0.5)
            self.consolidation.queue_for_consolidation(
                x.mean(dim=1).detach(),
                importance=importance,
            )
        
        # ============================================
        # OUTPUT GENERATION
        # ============================================
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # ============================================
        # LOSS COMPUTATION
        # ============================================
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # ============================================
        # COMPILE OUTPUT
        # ============================================
        output = {
            'logits': logits,
            'loss': loss,
        }
        
        if return_brain_state:
            output['brain_state'] = BrainState(
                dopamine_level=dopamine_level,
                arousal=self.current_arousal.item(),
                confidence=dacc_meta.get('confidence', 0.5),
                cognitive_load=dacc_meta.get('cognitive_load', 0.5),
                oscillation_phase=oscillation_phase or 0.0,
                active_modules=alloc_meta.get('per_module', {}).keys(),
            )
            output['metadata'] = {
                'thalamus': thalamus_meta,
                'cortex': cortex_meta,
                'working_memory': wm_meta,
                'hippocampus': hippocampus_meta,
                'semantic': semantic_meta,
                'dacc': dacc_meta,
                'dopamine': da_meta,
                'allocation': alloc_meta,
            }
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_striatum: bool = False,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            use_striatum: Use striatum for action selection
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop if too long
            if input_ids.shape[1] > self.config.max_sequence_length:
                input_ids = input_ids[:, -self.config.max_sequence_length:]
            
            # Forward pass
            outputs = self.forward(input_ids, store_memory=True)
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
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def consolidate(self, max_steps: int = 100) -> Dict:
        """
        Run memory consolidation (simulate sleep).
        
        Transfers episodic memories to semantic memory and
        applies synaptic homeostasis.
        """
        modules_for_homeostasis = [
            self.thalamus,
            self.working_memory,
            self.striatum,
            self.semantic_memory,
        ]
        
        return self.consolidation.full_consolidation_session(
            semantic_memory=self.semantic_memory,
            hippocampus=self.hippocampus,
            modules_for_homeostasis=modules_for_homeostasis,
            max_steps=max_steps,
        )
    
    def get_plasticity_stats(self) -> Dict[str, float]:
        """Get statistics on synaptic weights across modules."""
        stats = {}
        
        # Collect synaptic weights from all modules
        if hasattr(self.thalamus, 'synaptic_weights'):
            stats['thalamus_sw_mean'] = self.thalamus.synaptic_weights.mean().item()
        
        if hasattr(self.working_memory, 'synaptic_weights'):
            stats['working_memory_sw_mean'] = self.working_memory.synaptic_weights.mean().item()
        
        if hasattr(self.striatum, 'synaptic_weights'):
            stats['striatum_sw_mean'] = self.striatum.synaptic_weights.mean().item()
        
        if hasattr(self.semantic_memory, 'synaptic_weights'):
            valid_weights = self.semantic_memory.synaptic_weights[:max(1, self.semantic_memory.num_concepts.item())]
            stats['semantic_sw_mean'] = valid_weights.mean().item()
        
        return stats
    
    def get_energy_stats(self) -> Dict[str, float]:
        """Get energy/efficiency statistics."""
        return {
            'attention_efficiency': self.attention_controller.get_efficiency(),
            'thalamus_pass_rate': (
                self.thalamus.passed_inputs.float() / 
                (self.thalamus.total_inputs + 1)
            ).item(),
        }
