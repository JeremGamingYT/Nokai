"""
Nōkai Training Example with Neuromorphic Architecture

This script demonstrates how to:
1. Initialize the complete neuromorphic brain
2. Train with dopamine-modulated learning
3. Use memory consolidation ("sleep")
4. Monitor plasticity and efficiency
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time

from nokai import NokaiConfig, NeuromorphicBrain


def create_brain(preset: str = "mini") -> NeuromorphicBrain:
    """
    Create a neuromorphic brain with the specified preset.
    
    Presets:
        - nano: ~4M params, 200MB VRAM
        - micro: ~17M params, 500MB VRAM
        - mini: ~67M params, 2GB VRAM
        - base: ~268M params, 6GB VRAM (RTX 5070 target)
        - large: ~1B params, 16GB VRAM
    """
    config_methods = {
        "nano": NokaiConfig.nano,
        "micro": NokaiConfig.micro,
        "mini": NokaiConfig.mini,
        "base": NokaiConfig.base,
        "large": NokaiConfig.large,
    }
    
    config = config_methods[preset]()
    brain = NeuromorphicBrain(config)
    
    return brain


def train_step(
    brain: NeuromorphicBrain,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    reward_signal: Optional[float] = None,
) -> dict:
    """
    Perform one training step with neuromorphic features.
    
    Key differences from standard training:
        1. Dopamine modulates learning rate
        2. Memory systems store experiences
        3. Metacognition monitors uncertainty
        4. Synaptic plasticity is tracked
    """
    optimizer.zero_grad()
    
    input_ids = batch['input_ids']
    labels = batch.get('labels', input_ids)
    
    # Create reward tensor if provided
    reward = None
    if reward_signal is not None:
        reward = torch.tensor([reward_signal], device=input_ids.device)
    
    # Forward pass with memory storage
    outputs = brain(
        input_ids,
        labels=labels,
        reward=reward,
        store_memory=True,
        return_brain_state=True,
    )
    
    loss = outputs['loss']
    
    if loss is not None:
        # Get dopamine-modulated learning rate
        dopamine_mod = brain.dopamine_circuit.get_learning_modulation()
        
        # Scale gradients by dopamine (higher DA = faster learning)
        loss = loss / dopamine_mod  # Inverse because we're minimizing
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        
        optimizer.step()
    
    # Collect stats
    brain_state = outputs.get('brain_state')
    metadata = outputs.get('metadata', {})
    
    return {
        'loss': loss.item() if loss is not None else 0,
        'dopamine_level': brain_state.dopamine_level if brain_state else 0,
        'confidence': brain_state.confidence if brain_state else 0,
        'plasticity': brain.get_plasticity_stats(),
        'efficiency': brain.get_energy_stats(),
    }


def consolidation_phase(
    brain: NeuromorphicBrain,
    steps: int = 100,
) -> dict:
    """
    Run memory consolidation (simulates sleep).
    
    This phase:
        1. Transfers episodic → semantic
        2. Applies synaptic homeostasis
        3. Clears working memory
    """
    print("Starting consolidation (sleep simulation)...")
    brain.eval()
    
    stats = brain.consolidate(max_steps=steps)
    
    print(f"  Consolidated: {stats.get('total_consolidated', 0)} memories")
    print(f"  Pruned: {stats.get('total_pruned', 0)} weak memories")
    
    return stats


def training_loop(
    brain: NeuromorphicBrain,
    dataloader: DataLoader,
    num_epochs: int = 10,
    consolidation_interval: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Complete training loop with neuromorphic features.
    
    Features:
        - Dopamine-modulated learning
        - Periodic consolidation
        - Plasticity monitoring
        - Energy efficiency tracking
    """
    brain = brain.to(device)
    brain.train()
    
    optimizer = torch.optim.AdamW(
        brain.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )
    
    step = 0
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_start = time.time()
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Compute reward based on recent performance
            # (In practice, this could be based on task success)
            reward = None
            if step > 0 and len(epoch_losses) > 0:
                recent_avg = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                if epoch_losses[-1] < recent_avg:
                    reward = 0.1  # Positive RPE: better than expected
                else:
                    reward = -0.1  # Negative RPE: worse than expected
            
            # Training step
            stats = train_step(brain, batch, optimizer, reward)
            epoch_losses.append(stats['loss'])
            
            # Periodic consolidation
            if step > 0 and step % consolidation_interval == 0:
                consolidation_phase(brain, steps=50)
                brain.train()
            
            # Logging
            if step % 100 == 0:
                print(f"Step {step}:")
                print(f"  Loss: {stats['loss']:.4f}")
                print(f"  Dopamine: {stats['dopamine_level']:.3f}")
                print(f"  Confidence: {stats['confidence']:.3f}")
                print(f"  Efficiency: {stats['efficiency']}")
            
            step += 1
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Plasticity: {brain.get_plasticity_stats()}")
        
        # End-of-epoch consolidation
        consolidation_phase(brain, steps=100)
        brain.train()
    
    return brain


def demo():
    """
    Quick demonstration of the neuromorphic brain.
    """
    print("=" * 60)
    print("NŌKAI NEUROMORPHIC BRAIN DEMO")
    print("=" * 60)
    
    # Create brain
    brain = create_brain("nano")  # Small for demo
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brain = brain.to(device)
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    # Forward pass
    print("\n1. Forward Pass")
    outputs = brain(input_ids, store_memory=True, return_brain_state=True)
    
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Brain state:")
    print(f"    - Dopamine: {outputs['brain_state'].dopamine_level:.3f}")
    print(f"    - Confidence: {outputs['brain_state'].confidence:.3f}")
    print(f"    - Cognitive load: {outputs['brain_state'].cognitive_load:.3f}")
    
    # Check plasticity
    print("\n2. Plasticity Stats")
    plasticity = brain.get_plasticity_stats()
    for key, value in plasticity.items():
        print(f"  {key}: {value:.3f}")
    
    # Check efficiency
    print("\n3. Energy Efficiency")
    efficiency = brain.get_energy_stats()
    for key, value in efficiency.items():
        print(f"  {key}: {value:.3f}")
    
    # Consolidation
    print("\n4. Memory Consolidation")
    consolidation_phase(brain, steps=10)
    
    # Generation
    print("\n5. Text Generation")
    brain.eval()
    prompt = torch.randint(0, 1000, (1, 10), device=device)
    generated = brain.generate(prompt, max_new_tokens=20)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
