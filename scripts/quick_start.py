#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nōkai Quick Start - Fast Training Demo

This is a simplified training script for quick testing.
It uses a smaller dataset and fewer steps for rapid iteration.

Usage:
    python scripts/quick_start.py
    python scripts/quick_start.py --steps 100
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Fix Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def quick_train(steps: int = 100, preset: str = "nano"):
    """Quick training demo with synthetic data."""
    
    print("\n" + "="*60)
    print("NOKAI QUICK START")
    print("="*60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Model
    print(f"\nCreating {preset} model...")
    config = getattr(NokaiConfig, preset)()
    brain = NeuromorphicBrain(config)
    brain = brain.to(device)
    brain.train()
    
    params = sum(p.numel() for p in brain.parameters())
    print(f"Parameters: {params:,}")
    
    # Optimizer
    optimizer = AdamW(brain.parameters(), lr=1e-4)
    
    # Synthetic data for quick testing
    print(f"\nTraining for {steps} steps with synthetic data...")
    
    losses = []
    dopamine_levels = []
    
    iterator = range(steps)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Training")
    
    for step in iterator:
        # Generate random batch
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        
        # Compute reward (improvement = positive reward)
        reward = None
        if len(losses) > 0:
            improvement = losses[-1] - losses[-1] if len(losses) < 2 else losses[-2] - losses[-1]
            reward = torch.tensor([improvement * 10], device=device).clamp(-1, 1)
        
        # Forward
        outputs = brain(
            input_ids,
            labels=labels,
            reward=reward,
            store_memory=True,
            return_brain_state=True,
        )
        
        loss = outputs['loss']
        brain_state = outputs['brain_state']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        # Track
        losses.append(loss.item())
        dopamine_levels.append(brain_state.dopamine_level)
        
        # Log
        if (step + 1) % 20 == 0:
            avg_loss = sum(losses[-20:]) / len(losses[-20:])
            avg_da = sum(dopamine_levels[-20:]) / len(dopamine_levels[-20:])
            
            if HAS_TQDM:
                iterator.set_postfix(loss=f'{avg_loss:.4f}', DA=f'{avg_da:.2f}')
            else:
                print(f"Step {step+1}: loss={avg_loss:.4f}, DA={avg_da:.2f}")
        
        # Consolidation
        if (step + 1) % 50 == 0:
            print(f"\n  [Consolidation at step {step+1}]")
            brain.eval()
            stats = brain.consolidate(max_steps=10)
            brain.train()
            print(f"    Consolidated: {stats.get('total_consolidated', 0)}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss improvement: {losses[0] - losses[-1]:.4f}")
    print(f"  Final dopamine: {dopamine_levels[-1]:.3f}")
    print(f"  Plasticity: {brain.get_plasticity_stats()}")
    print(f"  Energy efficiency: {brain.get_energy_stats()}")
    
    # Test generation
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60)
    
    brain.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    
    with torch.no_grad():
        generated = brain.generate(prompt, max_new_tokens=20, temperature=0.8)
    
    print(f"  Generated {generated.shape[1]} tokens successfully!")
    
    # Save
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    torch.save(brain.state_dict(), str(checkpoint_dir / "brain_quickstart.pt"))
    print(f"\n  Model saved to: checkpoints/brain_quickstart.pt")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Full training: python scripts/train_wikipedia.py --preset mini")
    print("  2. Chat: python scripts/chat.py --checkpoint checkpoints/brain_quickstart.pt")
    print("="*60 + "\n")
    
    return brain


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--preset", type=str, default="nano", 
                       choices=["nano", "micro", "mini"])
    args = parser.parse_args()
    
    quick_train(args.steps, args.preset)
