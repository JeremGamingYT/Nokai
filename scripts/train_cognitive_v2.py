#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nōkai Cognitive Training Loop V2 - Complete Neuromorphic Training

=============================================================================
MAJOR IMPROVEMENTS OVER V1
=============================================================================

1. BPE TOKENIZATION (Semantic Understanding)
   - Replaces character-level tokenization with BPE subwords
   - Vocab size 32k-50k for concept-level understanding
   - The model now manipulates MEANINGS, not letters
   
2. HOMEOSTATIC DOPAMINE (True Reward Prediction Error)
   - Dopamine based on SURPRISE, not raw success
   - Hedonic adaptation prevents saturation at 1.0
   - Learning is modulated by genuine prediction errors

3. IMMEDIATE HEBBIAN LEARNING
   - Local plasticity during forward pass
   - "Neurons that fire together, wire together" - instantly
   - Complements backprop with biologically-plausible updates

=============================================================================
TRAINING ARCHITECTURE
=============================================================================

Each training step:
    1. Encode batch with BPE tokenizer
    2. Forward pass through NeuromorphicBrain
    3. DURING forward: Hebbian updates fire in cortical columns
    4. Compute loss from logits
    5. Compute reward = -loss_improvement (surprise-based)
    6. Update dopamine with homeostasis
    7. Backprop with dopamine-modulated learning rate
    8. Periodic consolidation ("sleep" phases)

This creates a hybrid learning system:
    - Global optimization via gradient descent
    - Local refinement via immediate Hebbian updates
    - Reward-relevant learning via dopamine gating

=============================================================================

Author: Nōkai Neuro-Engineering Team
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Nōkai imports
from nokai import NeuromorphicBrain, NokaiConfig
from nokai.learning import HebbianPlasticity, HebbianConfig, CorticalHebbianIntegrator

# Try to import tokenization
try:
    from nokai.tokenization import NokaiTokenizer, TokenizerConfig, create_tokenizer, HAS_TOKENIZERS
    USE_BPE = HAS_TOKENIZERS
except ImportError:
    USE_BPE = False
    print("[Warning] Tokenization module not available. Using fallback.")

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

# Datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[Error] datasets library required. Install with: pip install datasets")


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class TrainingConfigV2:
    """
    Enhanced training configuration.
    
    Biological Mapping:
        - hebbian_lr: Local plasticity strength (synaptic learning)
        - dopamine_target: Baseline motivation level
        - surprise_scale: How much RPE affects learning
        - consolidation_interval: "Sleep" frequency
    """
    # Model
    preset: str = "mini"
    
    # Training basics
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    
    # Sequence handling
    max_seq_length: int = 512
    gradient_accumulation: int = 1
    
    # BPE Tokenization
    vocab_size: int = 32000  # Target vocab size (actual may be smaller based on data)
    min_token_frequency: int = 2
    use_bpe: bool = True
    
    # Hebbian Learning
    hebbian_enabled: bool = True
    hebbian_lr: float = 0.001
    hebbian_interval: int = 1  # Apply every N steps (1 = every step)
    dopamine_gating: bool = True
    
    # Dopamine/Reward
    reward_scale: float = 10.0  # Scale loss improvement to reward
    surprise_learning_bonus: float = 0.5  # Extra LR boost for surprising events
    
    # Consolidation
    consolidation_interval: int = 500
    consolidation_steps: int = 50
    
    # Logging/Saving
    log_interval: int = 50
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    
    # Data - NEW: support for local file
    data_file: Optional[str] = None  # Path to local text file (overrides dataset)
    dataset: str = "wikitext"  # or "wikipedia" (used if data_file is None)
    dataset_config: str = "wikitext-2-v1"
    max_samples: Optional[int] = None
    
    # Hardware
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True


# ============================================
# TRAINING STATE
# ============================================

@dataclass
class TrainingState:
    """Complete state for checkpoint/resume."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    
    # History tracking
    losses: List[float] = field(default_factory=list)
    dopamine_history: List[float] = field(default_factory=list)
    surprise_history: List[float] = field(default_factory=list)
    hebbian_updates: int = 0
    consolidations: int = 0
    
    # Running averages
    avg_loss: float = 0.0
    avg_dopamine: float = 0.5
    avg_surprise: float = 0.0
    
    def update_averages(self, loss: float, dopamine: float, surprise: float):
        """Update exponential moving averages."""
        alpha = 0.01
        self.avg_loss = (1 - alpha) * self.avg_loss + alpha * loss if self.avg_loss > 0 else loss
        self.avg_dopamine = (1 - alpha) * self.avg_dopamine + alpha * dopamine
        self.avg_surprise = (1 - alpha) * self.avg_surprise + alpha * surprise
    
    def save(self, path: str):
        """Save state to file."""
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'losses': self.losses[-1000:],
            'dopamine_history': self.dopamine_history[-1000:],
            'surprise_history': self.surprise_history[-1000:],
            'hebbian_updates': self.hebbian_updates,
            'consolidations': self.consolidations,
            'avg_loss': self.avg_loss,
            'avg_dopamine': self.avg_dopamine,
            'avg_surprise': self.avg_surprise,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "TrainingState":
        """Load state from file."""
        data = torch.load(path, weights_only=False)
        state = cls()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


# ============================================
# DATASET
# ============================================

class TextDataset(Dataset):
    """
    Text dataset with BPE tokenization.
    
    Handles both character-level fallback and proper BPE tokenization.
    """
    
    def __init__(
        self,
        tokenizer,
        texts: List[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [t for t in texts if len(t) > 50]
        
        print(f"[Dataset] Loaded {len(self.texts)} valid texts")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            # BPE tokenizer
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )
        else:
            # Fallback character tokenizer
            tokens = [self.tokenizer.bos_token_id]
            for char in text[:self.max_length - 2]:
                tokens.append(self.tokenizer.char_to_id.get(char, self.tokenizer.unk_token_id))
            tokens.append(self.tokenizer.eos_token_id)
        
        # Pad or truncate
        if len(tokens) > self.max_length:
            # Random crop for variety
            start = torch.randint(0, len(tokens) - self.max_length, (1,)).item()
            tokens = tokens[start:start + self.max_length]
        elif len(tokens) < self.max_length:
            # Pad
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'input_ids': tokens,
            'labels': tokens.clone(),
        }


# ============================================
# FALLBACK TOKENIZER
# ============================================

class FallbackTokenizer:
    """Simple character-level tokenizer as fallback."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.char_to_id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.next_id = 4
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def train(self, texts: List[str]):
        """Build vocab from texts."""
        for text in texts:
            for char in text:
                if char not in self.char_to_id and self.next_id < self.vocab_size:
                    self.char_to_id[char] = self.next_id
                    self.id_to_char[self.next_id] = char
                    self.next_id += 1
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to IDs."""
        ids = [self.bos_token_id]
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode IDs to text."""
        chars = []
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(self.id_to_char.get(i, '?'))
        return ''.join(chars)
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'char_to_id': self.char_to_id, 'vocab_size': self.vocab_size}, f)
    
    @classmethod
    def load(cls, path: str) -> "FallbackTokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(data['vocab_size'])
        tok.char_to_id = data['char_to_id']
        tok.id_to_char = {int(v): k for k, v in data['char_to_id'].items()}
        tok.next_id = max(tok.id_to_char.keys()) + 1
        return tok


# ============================================
# COGNITIVE TRAINER
# ============================================

class CognitiveTrainer:
    """
    Enhanced trainer with immediate Hebbian learning.
    
    Combines three learning systems:
        1. Global: Backpropagation (gradient descent)
        2. Local: Hebbian plasticity (immediate, in forward pass)
        3. Reward: Dopamine modulation (gates learning rate)
    
    Biological Parallel:
        The brain uses multiple learning mechanisms in parallel:
        - Synaptic plasticity (Hebbian) for immediate learning
        - Dopamine signals for reward-based modulation
        - Sleep consolidation for long-term memory
    """
    
    def __init__(
        self,
        brain: NeuromorphicBrain,
        config: TrainingConfigV2,
    ):
        self.brain = brain.to(config.device)
        self.config = config
        self.device = config.device
        
        # Optimizer
        self.optimizer = AdamW(
            brain.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler (set up later with total steps)
        self.scheduler = None
        
        # Mixed precision
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Hebbian integrators for cortical layers
        self.hebbian_integrators = []
        if config.hebbian_enabled:
            self._setup_hebbian_integrators()
        
        # State tracking
        self.state = TrainingState()
        self.prev_loss = None
        
        print(f"\n[CognitiveTrainer] Initialized")
        print(f"  Device: {config.device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Hebbian Learning: {config.hebbian_enabled}")
        print(f"  Dopamine Gating: {config.dopamine_gating}")
    
    def _setup_hebbian_integrators(self):
        """Set up Hebbian learning for cortical layers."""
        # Find feedforward layers in cortex
        for layer in self.brain.cortex.layers:
            for column in layer.columns:
                for ff in column.feedforward:
                    integrator = CorticalHebbianIntegrator(
                        existing_layer=nn.Linear(ff.in_features, ff.out_features),
                        hebbian_lr=self.config.hebbian_lr,
                        dopamine_gating=self.config.dopamine_gating,
                    )
                    # Copy weights (reference, not clone)
                    integrator.layer.weight = ff.weight
                    if ff.bias is not None:
                        integrator.layer.bias = ff.bias
                    self.hebbian_integrators.append(integrator)
        
        print(f"  Hebbian Integrators: {len(self.hebbian_integrators)}")
    
    def setup_scheduler(self, total_steps: int):
        """Set up learning rate scheduler."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )
    
    def compute_reward(self, current_loss: float) -> Tuple[torch.Tensor, float]:
        """
        Compute reward signal based on loss improvement.
        
        Biological Parallel:
            Reward = Prediction Error
            We expected a certain loss, getting less is positive surprise.
            Getting more is negative surprise (punishment).
        
        Returns:
            reward: Tensor for dopamine circuit
            surprise: Absolute prediction error
        """
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return torch.zeros(1, device=self.device), 0.0
        
        # Improvement = positive reward
        improvement = self.prev_loss - current_loss
        
        # Scale to reasonable range
        reward = improvement * self.config.reward_scale
        reward = max(-1.0, min(1.0, reward))
        
        # Surprise = absolute prediction error
        surprise = abs(improvement)
        
        self.prev_loss = current_loss
        
        return torch.tensor([reward], device=self.device), surprise
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with integrated Hebbian learning.
        
        Architecture:
            1. Forward pass (Hebbian updates fire during this)
            2. Compute loss and reward
            3. Update dopamine with homeostasis
            4. Modulate learning rate based on dopamine
            5. Backprop with gradient clipping
        """
        self.brain.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # =====================================
        # 1. GET PENDING REWARD FROM PREVIOUS STEP
        # =====================================
        # The reward is based on loss improvement from previous step
        # First step has no reward (None triggers default behavior in brain)
        pending_reward = getattr(self, '_pending_reward', None)
        
        # =====================================
        # 2. FORWARD PASS (with reward from previous step)
        # =====================================
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            outputs = self.brain(
                input_ids,
                labels=labels,
                reward=pending_reward,  # Reward from previous step
                store_memory=True,
                return_brain_state=True,
            )
            loss = outputs['loss']
            brain_state = outputs.get('brain_state', None)
        
        current_loss = loss.item()
        
        # =====================================
        # 3. COMPUTE REWARD FOR NEXT STEP
        # =====================================
        # This reward will be used in the NEXT train_step
        reward_tensor, surprise = self.compute_reward(current_loss)
        self._pending_reward = reward_tensor  # Store for next step
        
        # =====================================
        # 4. GET DOPAMINE STATE FROM BRAIN
        # =====================================
        # The brain already updates dopamine during forward when reward is passed
        if brain_state is not None:
            dopamine_level = brain_state.dopamine_level
        else:
            dopamine_level = 0.5
        
        # Get additional dopamine metadata if available
        da_meta = outputs.get('metadata', {}).get('dopamine', {})
        if not da_meta:
            da_meta = {'habituation': 0.0, 'rpe': 0.0}
        
        # =====================================
        # 5. HEBBIAN LEARNING (Immediate)
        # =====================================
        if self.config.hebbian_enabled and self.state.step % self.config.hebbian_interval == 0:
            # Apply Hebbian updates to all cortical layers
            for layer in self.brain.cortex.layers:
                for column in layer.columns:
                    # Get stored activations
                    if hasattr(column, 'pre_activations') and hasattr(column, 'post_activations'):
                        for i, ff in enumerate(column.feedforward):
                            if i < len(column.pre_activations) - 1:
                                pre = column.pre_activations[i]
                                post = column.post_activations[i + 1]
                                
                                # Convert to float32 and ensure correct shape
                                pre = pre.float()
                                post = post.float()
                                if pre.dim() == 1:
                                    pre = pre.unsqueeze(0)
                                if post.dim() == 1:
                                    post = post.unsqueeze(0)
                                
                                # Compute and apply Hebbian update
                                with torch.no_grad():
                                    mask = getattr(ff, 'mask', None)
                                    hebbian_delta = self._compute_hebbian_update(
                                        ff.weight.float(),
                                        pre,
                                        post,
                                        dopamine_level,
                                        mask,
                                    )
                                    ff.weight.data.add_(hebbian_delta.to(ff.weight.dtype))
                                    self.state.hebbian_updates += 1
        
        # =====================================
        # 5. MODULATE LEARNING RATE
        # =====================================
        da_modulation = self.brain.dopamine_circuit.get_learning_modulation()
        
        # Add surprise bonus
        surprise_bonus = min(self.config.surprise_learning_bonus, surprise * 0.5)
        effective_lr = self.config.learning_rate * da_modulation * (1 + surprise_bonus)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = effective_lr
        
        # =====================================
        # 6. BACKWARD PASS
        # =====================================
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.brain.parameters(),
                self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.brain.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        # =====================================
        # 7. UPDATE STATE
        # =====================================
        self.state.step += 1
        self.state.losses.append(current_loss)
        self.state.dopamine_history.append(dopamine_level)
        self.state.surprise_history.append(surprise)
        self.state.update_averages(current_loss, dopamine_level, surprise)
        
        return {
            'loss': current_loss,
            'dopamine': dopamine_level,
            'surprise': surprise,
            'lr': effective_lr,
            'da_modulation': da_modulation,
            'habituation': da_meta.get('habituation', 0.0),
            'rpe': da_meta.get('rpe', 0.0),
        }
    
    def _compute_hebbian_update(
        self,
        weight: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        dopamine: float,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Hebbian weight update.
        
        Implements Oja's rule with dopamine gating:
            Δw = η · DA · (post ⊗ pre - α · post² · w)
        """
        lr = self.config.hebbian_lr
        
        # Average over batch if needed
        if pre.dim() > 1:
            pre = pre.mean(0)
        if post.dim() > 1:
            post = post.mean(0)
        
        # Hebbian term: post ⊗ pre
        hebbian = torch.outer(post, pre)
        
        # Oja's normalization
        oja_decay = 0.01 * (post ** 2).unsqueeze(1) * weight
        
        # Combine with dopamine gating
        if self.config.dopamine_gating:
            gate = max(0, dopamine - 0.3) / 0.7  # Only learn with DA > 0.3
        else:
            gate = 1.0
        
        delta = lr * gate * (hebbian - oja_decay)
        
        # Apply sparse mask if available
        if mask is not None:
            delta = delta * mask
        
        return delta.clamp(-1.0, 1.0)
    
    def consolidate(self) -> Dict:
        """
        Run memory consolidation ("sleep" phase).
        
        Biological Parallel:
            During sleep, the brain:
            - Replays recent memories
            - Transfers important memories to long-term storage
            - Applies synaptic homeostasis (downscaling)
        """
        self.brain.eval()
        stats = self.brain.consolidate(max_steps=self.config.consolidation_steps)
        self.brain.train()
        self.state.consolidations += 1
        return stats
    
    def save_checkpoint(self, path: str):
        """Save complete checkpoint."""
        torch.save({
            'model_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_state': {
                'step': self.state.step,
                'epoch': self.state.epoch,
                'best_loss': self.state.best_loss,
                'avg_loss': self.state.avg_loss,
            },
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.brain.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        ts = checkpoint.get('training_state', {})
        self.state.step = ts.get('step', 0)
        self.state.epoch = ts.get('epoch', 0)
        self.state.best_loss = ts.get('best_loss', float('inf'))


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def train_cognitive_v2(config: TrainingConfigV2):
    """Main training function with all cognitive enhancements."""
    
    print("\n" + "=" * 70)
    print("NŌKAI COGNITIVE TRAINING V2.1 - ALIGNED BRAIN")
    print("=" * 70)
    print(f"  Preset: {config.preset}")
    print(f"  BPE Tokenization: {config.use_bpe and USE_BPE}")
    print(f"  Hebbian Learning: {config.hebbian_enabled}")
    print(f"  Dopamine Gating: {config.dopamine_gating}")
    print("=" * 70)
    
    # =====================================
    # SETUP
    # =====================================
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # =====================================
    # STEP 1: LOAD DATA FIRST
    # =====================================
    print(f"\n[1/5] Loading data...")
    
    texts = []
    
    # Option A: Load from local file
    if config.data_file and Path(config.data_file).exists():
        print(f"  Loading from file: {config.data_file}")
        with open(config.data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split by double newlines (story separator)
        raw_texts = content.split('\n\n')
        texts = [t.strip() for t in raw_texts if len(t.strip()) > 50]
        print(f"  Loaded {len(texts)} texts from file")
    
    # Option B: Load from HuggingFace
    else:
        print(f"  Loading from HuggingFace ({config.dataset})...")
        try:
            if config.dataset == "wikipedia":
                dataset = load_dataset(
                    "wikipedia",
                    "20231101.en",
                    split="train",
                    trust_remote_code=True,
                )
            elif config.dataset == "tinystories":
                dataset = load_dataset(
                    "roneneldan/TinyStories",
                    split="train",
                )
            else:
                dataset = load_dataset(
                    "wikitext",
                    config.dataset_config,
                    split="train",
                )
        except Exception as e:
            print(f"  Dataset load failed: {e}")
            print("  Falling back to wikitext-2...")
            dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        
        # Extract texts
        text_key = 'text' if 'text' in dataset.column_names else 'content'
        texts = [item[text_key] for item in dataset if len(item.get(text_key, '')) > 50]
        print(f"  Loaded {len(texts)} texts from dataset")
    
    if len(texts) < 100:
        print(f"\n⚠️  WARNING: Only {len(texts)} texts loaded!")
        print("    This is insufficient for proper training.")
        print("    Run: python scripts/download_data.py --split 0.05")
        print("    Then: python scripts/train_cognitive_v2.py --data_file data/tinystories.txt")
    
    # =====================================
    # STEP 2: TRAIN TOKENIZER ON DATA
    # =====================================
    print(f"\n[2/5] Setting up tokenizer (DATA-ALIGNED)...")
    
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    actual_vocab_size = config.vocab_size  # Will be updated after training
    
    if USE_BPE and config.use_bpe:
        if tokenizer_path.exists():
            print("  Loading existing BPE tokenizer...")
            tokenizer = NokaiTokenizer.load(str(tokenizer_path))
            actual_vocab_size = tokenizer.vocab_size
            print(f"  Tokenizer vocab: {actual_vocab_size}")
        else:
            print("  Training NEW BPE tokenizer on data...")
            sample_texts = texts[:min(100000, len(texts))]
            tokenizer = NokaiTokenizer.train(sample_texts, TokenizerConfig(
                vocab_size=config.vocab_size,
                min_frequency=config.min_token_frequency,
            ))
            tokenizer.save(str(tokenizer_path))
            actual_vocab_size = tokenizer.vocab_size
            print(f"  Tokenizer trained! Vocab size: {actual_vocab_size}")
    else:
        if tokenizer_path.exists():
            print("  Loading existing character tokenizer...")
            tokenizer = FallbackTokenizer.load(str(tokenizer_path))
            actual_vocab_size = len(tokenizer.char_to_id)
        else:
            print("  Training character tokenizer...")
            tokenizer = FallbackTokenizer(config.vocab_size)
            tokenizer.train(texts[:10000])
            tokenizer.save(str(tokenizer_path))
            actual_vocab_size = len(tokenizer.char_to_id)
        print(f"  Character tokenizer vocab: {actual_vocab_size}")
    
    # =====================================
    # STEP 3: CREATE BRAIN WITH ALIGNED VOCAB
    # =====================================
    print(f"\n[3/5] Creating NeuromorphicBrain ({config.preset})...")
    print(f"  ⚡ ALIGNED: Using tokenizer vocab_size = {actual_vocab_size}")
    
    brain_config = getattr(NokaiConfig, config.preset)()
    brain_config.max_sequence_length = config.max_seq_length
    
    # CRITICAL: Use tokenizer's ACTUAL vocab size, not the config target
    brain_config.vocab_size = actual_vocab_size
    
    brain = NeuromorphicBrain(brain_config)
    brain = brain.to(device)
    
    param_count = sum(p.numel() for p in brain.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Embedding dim: {brain_config.embedding_dim}")
    print(f"  Vocab size: {brain_config.vocab_size} (ALIGNED with tokenizer)")
    
    # =====================================
    # STEP 4: CREATE DATASET AND DATALOADER
    # =====================================
    print(f"\n[4/5] Creating dataset...")
    
    train_dataset = TextDataset(tokenizer, texts, config.max_seq_length)
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # =====================================
    # STEP 5: INITIALIZE TRAINER
    # =====================================
    print(f"\n[5/5] Initializing trainer...")
    
    trainer = CognitiveTrainer(brain, config)
    
    total_steps = len(dataloader) * config.epochs
    trainer.setup_scheduler(total_steps)
    
    # Auto-adjust intervals
    consolidation_interval = min(config.consolidation_interval, max(20, total_steps // 5))
    save_interval = min(config.save_interval, max(10, total_steps // 3))
    
    # =====================================
    # TRAINING LOOP
    # =====================================
    print(f"\n[TRAINING] Starting...")
    print(f"  Epochs: {config.epochs}")
    print(f"  Steps per epoch: {len(dataloader)}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Consolidation every: {consolidation_interval} steps")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_losses = []
        trainer.state.epoch = epoch
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}") if HAS_TQDM else dataloader
        
        for batch in progress:
            metrics = trainer.train_step(batch)
            epoch_losses.append(metrics['loss'])
            
            # Logging
            if trainer.state.step % config.log_interval == 0:
                if HAS_TQDM:
                    progress.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'DA': f"{metrics['dopamine']:.2f}",
                        'surp': f"{metrics['surprise']:.3f}",
                        'hab': f"{metrics['habituation']:.2f}",
                    })
                else:
                    print(f"Step {trainer.state.step}: loss={metrics['loss']:.4f}, "
                          f"DA={metrics['dopamine']:.2f}, surprise={metrics['surprise']:.3f}")
            
            # Consolidation
            if trainer.state.step > 0 and trainer.state.step % consolidation_interval == 0:
                print(f"\n{'='*50}")
                print(f"CONSOLIDATION (Sleep) - Step {trainer.state.step}")
                print(f"{'='*50}")
                
                stats = trainer.consolidate()
                
                print(f"  Consolidated: {stats.get('total_consolidated', 0)}")
                print(f"  Pruned: {stats.get('total_pruned', 0)}")
                print(f"  Hebbian updates: {trainer.state.hebbian_updates}")
                print(f"  Avg Dopamine: {trainer.state.avg_dopamine:.3f}")
                print(f"  Avg Surprise: {trainer.state.avg_surprise:.3f}")
                print(f"{'='*50}\n")
            
            # Save checkpoint
            if trainer.state.step > 0 and trainer.state.step % save_interval == 0:
                trainer.save_checkpoint(str(checkpoint_dir / "brain_latest.pt"))
                trainer.state.save(str(checkpoint_dir / "training_state.pt"))
                
                avg_loss = sum(epoch_losses[-100:]) / max(1, len(epoch_losses[-100:]))
                if avg_loss < trainer.state.best_loss:
                    trainer.state.best_loss = avg_loss
                    trainer.save_checkpoint(str(checkpoint_dir / "brain_best.pt"))
                    print(f"  New best loss: {avg_loss:.4f}")
        
        # Epoch summary
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"\n Epoch {epoch+1} Complete:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Dopamine: {trainer.state.avg_dopamine:.3f}")
        print(f"  Surprise: {trainer.state.avg_surprise:.3f}")
        print(f"  Hebbian Updates: {trainer.state.hebbian_updates:,}")
        
        # Save epoch checkpoint
        trainer.save_checkpoint(str(checkpoint_dir / f"brain_epoch_{epoch+1}.pt"))
    
    # =====================================
    # FINAL SUMMARY
    # =====================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total steps: {trainer.state.step:,}")
    print(f"  Hebbian updates: {trainer.state.hebbian_updates:,}")
    print(f"  Consolidations: {trainer.state.consolidations}")
    print(f"  Best loss: {trainer.state.best_loss:.4f}")
    print(f"  Final Dopamine: {trainer.state.avg_dopamine:.3f}")
    print(f"  Model saved to: {checkpoint_dir}/brain_best.pt")
    print("=" * 70)
    
    return brain, tokenizer


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Nōkai Cognitive Training V2")
    
    # Model
    parser.add_argument("--preset", type=str, default="mini",
                       choices=["nano", "micro", "mini", "base", "large"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    
    # Cognitive features
    parser.add_argument("--no_hebbian", action="store_true", help="Disable Hebbian learning")
    parser.add_argument("--no_dopamine_gating", action="store_true", help="Disable dopamine gating")
    parser.add_argument("--hebbian_lr", type=float, default=0.001)
    
    # Tokenization
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--no_bpe", action="store_true", help="Use character-level instead of BPE")
    
    # Data
    parser.add_argument("--data_file", type=str, default=None,
                       help="Path to local text file (overrides --dataset)")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "wikipedia", "tinystories"])
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Consolidation
    parser.add_argument("--consolidation_interval", type=int, default=500)
    
    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = TrainingConfigV2(
        preset=args.preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        hebbian_enabled=not args.no_hebbian,
        dopamine_gating=not args.no_dopamine_gating,
        hebbian_lr=args.hebbian_lr,
        vocab_size=args.vocab_size,
        use_bpe=not args.no_bpe,
        data_file=args.data_file,
        dataset=args.dataset,
        max_samples=args.max_samples,
        consolidation_interval=args.consolidation_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    
    train_cognitive_v2(config)


if __name__ == "__main__":
    main()
