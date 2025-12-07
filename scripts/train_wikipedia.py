#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nōkai Neuromorphic Brain - Complete Training Script with Wikipedia

This script trains the neuromorphic brain on Wikipedia data with:
1. Dopamine-modulated learning
2. Periodic memory consolidation (sleep phases)
3. Synaptic plasticity tracking
4. Energy-efficient processing

Usage:
    python scripts/train_wikipedia.py --preset mini --epochs 10
    python scripts/train_wikipedia.py --preset base --epochs 50 --batch_size 4
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig

# Optional imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("ERROR: datasets library required. Install: pip install datasets")
    sys.exit(1)

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False
    print("Note: Using simple tokenizer (install tokenizers for better results)")


# ============================================
# CONFIGURATION
# ============================================

DEFAULT_CONFIG = {
    "preset": "mini",           # nano, micro, mini, base, large
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "max_seq_length": 512,
    "gradient_accumulation": 1,
    "consolidation_interval": 500,   # Steps between sleep phases
    "consolidation_steps": 50,       # Steps per sleep phase
    "save_interval": 1000,
    "log_interval": 50,
    "wikipedia_lang": "en",
    "wikipedia_date": "20231101",
    "max_samples": None,            # None = use all
    "seed": 42,
}


# ============================================
# SIMPLE TOKENIZER (fallback)
# ============================================

class SimpleTokenizer:
    """Simple character-level tokenizer as fallback."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize with special tokens
        self.char_to_id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.id_to_char = {0: '<pad>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.next_id = 4
    
    def train(self, texts: List[str]):
        """Build vocabulary from texts."""
        for text in texts:
            for char in text:
                if char not in self.char_to_id and self.next_id < self.vocab_size:
                    self.char_to_id[char] = self.next_id
                    self.id_to_char[self.next_id] = char
                    self.next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ids = [self.bos_token_id]
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for id in ids:
            if id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            chars.append(self.id_to_char.get(id, '?'))
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': {int(k): v for k, v in self.id_to_char.items()},
                'vocab_size': self.vocab_size,
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(data['vocab_size'])
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        return tokenizer


# ============================================
# DATASET
# ============================================

class WikipediaDataset(Dataset):
    """Wikipedia dataset for training."""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        lang: str = "en",
        date: str = "20231101",
        max_samples: Optional[int] = None,
        cache_dir: str = "./data/wikipedia",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading Wikipedia ({lang}, {date})...")
        print("This may take a while on first run (downloading ~20GB)...")
        
        try:
            # Try loading Wikipedia
            dataset = load_dataset(
                "wikipedia",
                f"{date}.{lang}",
                split=split,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Wikipedia download failed: {e}")
            print("Falling back to smaller dataset...")
            # Fallback to smaller dataset
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-v1",
                split="train",
                cache_dir=str(self.cache_dir),
            )
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.texts = []
        print("Processing texts...")
        
        # Extract text
        text_key = 'text' if 'text' in dataset.column_names else 'content'
        
        for item in tqdm(dataset, disable=not HAS_TQDM, desc="Loading"):
            text = item.get(text_key, "")
            if len(text) > 100:  # Skip very short texts
                self.texts.append(text)
        
        print(f"Loaded {len(self.texts)} documents")
        
        # Train tokenizer if needed
        if isinstance(tokenizer, SimpleTokenizer) and tokenizer.next_id < 1000:
            print("Training tokenizer on corpus...")
            sample_texts = self.texts[:min(10000, len(self.texts))]
            tokenizer.train(sample_texts)
            print(f"Vocabulary size: {tokenizer.next_id}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            # Random crop for variety
            start = torch.randint(0, len(tokens) - self.max_length, (1,)).item()
            tokens = tokens[start:start + self.max_length]
        else:
            # Pad
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'input_ids': tokens,
            'labels': tokens.clone(),
        }


# ============================================
# TRAINING UTILITIES
# ============================================

def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    return device


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_reward(loss: float, prev_loss: float) -> float:
    """
    Compute reward signal for dopamine circuit.
    
    Positive reward if loss improved, negative otherwise.
    """
    if prev_loss is None:
        return 0.0
    
    improvement = prev_loss - loss
    # Normalize to [-1, 1] range
    reward = max(-1.0, min(1.0, improvement * 10))
    return reward


class TrainingState:
    """Track training progress."""
    
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.losses = []
        self.dopamine_history = []
        self.plasticity_history = []
        self.consolidations = 0
    
    def save(self, path: str):
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'losses': self.losses[-1000:],  # Keep last 1000
            'dopamine_history': self.dopamine_history[-1000:],
            'consolidations': self.consolidations,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingState':
        state = cls()
        data = torch.load(path)
        state.step = data['step']
        state.epoch = data['epoch']
        state.best_loss = data['best_loss']
        state.losses = data['losses']
        state.dopamine_history = data['dopamine_history']
        state.consolidations = data['consolidations']
        return state


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def train(args):
    """Main training function."""
    
    print("\n" + "="*60)
    print("NOKAI NEUROMORPHIC BRAIN - TRAINING")
    print("="*60)
    
    # Setup
    torch.manual_seed(args.seed)
    device = setup_device()
    
    # Create directories
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # ============================================
    # MODEL
    # ============================================
    print(f"\nCreating NeuromorphicBrain ({args.preset} preset)...")
    
    config_methods = {
        "nano": NokaiConfig.nano,
        "micro": NokaiConfig.micro,
        "mini": NokaiConfig.mini,
        "base": NokaiConfig.base,
        "large": NokaiConfig.large,
    }
    
    config = config_methods[args.preset]()
    config.max_sequence_length = args.max_seq_length
    
    brain = NeuromorphicBrain(config)
    brain = brain.to(device)
    
    print(f"Parameters: {count_parameters(brain):,}")
    
    # ============================================
    # TOKENIZER
    # ============================================
    tokenizer_path = checkpoints_dir / "tokenizer.json"
    
    if tokenizer_path.exists():
        print("Loading existing tokenizer...")
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
    else:
        print("Creating new tokenizer...")
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # ============================================
    # DATASET
    # ============================================
    dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        split="train",
        lang=args.wikipedia_lang,
        date=args.wikipedia_date,
        max_samples=args.max_samples,
    )
    
    # Save tokenizer
    tokenizer.save(str(tokenizer_path))
    
    # Save config for inference
    config_path = checkpoints_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'preset': args.preset,
            'vocab_size': config.vocab_size,
            'embedding_dim': config.embedding_dim,
            'max_sequence_length': config.max_sequence_length,
            'num_columns': config.num_columns,
        }, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # ============================================
    # OPTIMIZER & SCHEDULER
    # ============================================
    optimizer = AdamW(
        brain.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # ============================================
    # TRAINING STATE
    # ============================================
    state_path = checkpoints_dir / "training_state.pt"
    if state_path.exists() and not args.fresh_start:
        print("Resuming from checkpoint...")
        state = TrainingState.load(str(state_path))
    else:
        state = TrainingState()
    
    # Load model checkpoint if exists
    model_path = checkpoints_dir / "brain_latest.pt"
    if model_path.exists() and not args.fresh_start:
        print("Loading model weights...")
        brain.load_state_dict(torch.load(str(model_path), map_location=device))
    
    # ============================================
    # TRAINING LOOP
    # ============================================
    
    # Auto-adjust intervals for small datasets
    effective_consolidation_interval = min(args.consolidation_interval, max(20, total_steps // 5))
    effective_save_interval = min(args.save_interval, max(10, total_steps // 3))
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Consolidation every: {effective_consolidation_interval} steps")
    print(f"  Saving every: {effective_save_interval} steps")
    print("="*60 + "\n")
    
    brain.train()
    prev_loss = None
    accum_loss = 0.0
    accum_steps = 0
    
    start_time = time.time()
    
    for epoch in range(state.epoch, args.epochs):
        epoch_losses = []
        epoch_start = time.time()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") if HAS_TQDM else dataloader
        
        for batch_idx, batch in enumerate(progress):
            state.step += 1
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Compute reward based on previous loss (for dopamine)
            reward = None
            if prev_loss is not None:
                reward_value = compute_reward(accum_loss / max(1, accum_steps), prev_loss)
                reward = torch.tensor([reward_value], device=device)
            
            # Forward pass
            outputs = brain(
                input_ids,
                labels=labels,
                reward=reward,
                store_memory=True,
                return_brain_state=True,
            )
            
            loss = outputs['loss']
            brain_state = outputs['brain_state']
            
            # Backward pass
            loss = loss / args.gradient_accumulation
            loss.backward()
            
            # Accumulate
            accum_loss += loss.item() * args.gradient_accumulation
            accum_steps += 1
            
            # Optimizer step
            if state.step % args.gradient_accumulation == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                
                # Dopamine-modulated learning rate
                da_mod = brain.dopamine_circuit.get_learning_modulation()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate * da_mod
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update prev_loss
                prev_loss = accum_loss / accum_steps
                accum_loss = 0.0
                accum_steps = 0
            
            # Track
            epoch_losses.append(loss.item() * args.gradient_accumulation)
            state.losses.append(loss.item() * args.gradient_accumulation)
            state.dopamine_history.append(brain_state.dopamine_level)
            
            # Logging
            if state.step % args.log_interval == 0:
                avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                
                if HAS_TQDM:
                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'DA': f'{brain_state.dopamine_level:.2f}',
                        'conf': f'{brain_state.confidence:.2f}',
                    })
                else:
                    print(f"Step {state.step}: loss={avg_loss:.4f}, DA={brain_state.dopamine_level:.2f}")
            
            # ============================================
            # CONSOLIDATION (Sleep Phase)
            # ============================================
            if state.step > 0 and state.step % effective_consolidation_interval == 0:
                print(f"\n{'='*40}")
                print(f"CONSOLIDATION PHASE (Sleep) - Step {state.step}")
                print(f"{'='*40}")
                
                brain.eval()
                consolidation_stats = brain.consolidate(max_steps=args.consolidation_steps)
                brain.train()
                
                print(f"  Memories consolidated: {consolidation_stats.get('total_consolidated', 0)}")
                print(f"  Weak memories pruned: {consolidation_stats.get('total_pruned', 0)}")
                print(f"  Plasticity stats: {brain.get_plasticity_stats()}")
                print(f"{'='*40}\n")
                
                state.consolidations += 1
            
            # ============================================
            # SAVE CHECKPOINT
            # ============================================
            if state.step > 0 and state.step % effective_save_interval == 0:
                avg_loss = sum(epoch_losses[-100:]) / max(1, len(epoch_losses[-100:]))
                
                # Save latest
                torch.save(brain.state_dict(), str(checkpoints_dir / "brain_latest.pt"))
                state.save(str(state_path))
                
                # Save best
                if avg_loss < state.best_loss:
                    state.best_loss = avg_loss
                    torch.save(brain.state_dict(), str(checkpoints_dir / "brain_best.pt"))
                    print(f"  New best loss: {avg_loss:.4f}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_time/60:.1f} min")
        print(f"  Dopamine: {brain_state.dopamine_level:.3f}")
        print(f"  Plasticity: {brain.get_plasticity_stats()}")
        print(f"  Energy efficiency: {brain.get_energy_stats()}")
        
        state.epoch = epoch + 1
        
        # Update best loss at end of epoch
        if avg_epoch_loss < state.best_loss:
            state.best_loss = avg_epoch_loss
            torch.save(brain.state_dict(), str(checkpoints_dir / "brain_best.pt"))
            print(f"  [NEW BEST LOSS: {avg_epoch_loss:.4f}]")
        
        # Save latest after each epoch
        torch.save(brain.state_dict(), str(checkpoints_dir / "brain_latest.pt"))
        state.save(str(state_path))
        
        # Save epoch checkpoint
        torch.save(
            brain.state_dict(), 
            str(checkpoints_dir / f"brain_epoch_{epoch+1}.pt")
        )
    
    # ============================================
    # FINAL SAVE
    # ============================================
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Total steps: {state.step:,}")
    print(f"  Consolidations: {state.consolidations}")
    print(f"  Best loss: {state.best_loss:.4f}")
    print(f"  Model saved to: {checkpoints_dir}/brain_best.pt")
    print("="*60)
    
    # Save final
    torch.save(brain.state_dict(), str(checkpoints_dir / "brain_final.pt"))
    state.save(str(state_path))
    
    return brain, tokenizer


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Train Nokai Neuromorphic Brain")
    
    # Model
    parser.add_argument("--preset", type=str, default="mini",
                       choices=["nano", "micro", "mini", "base", "large"],
                       help="Model size preset")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    
    # Bio-inspired
    parser.add_argument("--consolidation_interval", type=int, default=500,
                       help="Steps between sleep/consolidation phases")
    parser.add_argument("--consolidation_steps", type=int, default=50,
                       help="Memory consolidation steps per phase")
    
    # Data
    parser.add_argument("--wikipedia_lang", type=str, default="en")
    parser.add_argument("--wikipedia_date", type=str, default="20231101")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit dataset size (for testing)")
    
    # Checkpoints
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--fresh_start", action="store_true",
                       help="Ignore existing checkpoints")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
