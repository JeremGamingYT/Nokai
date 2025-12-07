#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nokai Brain Test Script - QA Validation

This script validates the entire neuromorphic architecture:
1. Module initialization
2. Tensor dimension flow
3. Forward/backward pass
4. Memory systems
5. Consolidation
6. Generation

Run: python test_brain.py
"""

import sys
import os
import torch
import traceback
from typing import Dict, List, Tuple

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Test counters
TESTS_PASSED = 0
TESTS_FAILED = 0
FAILURES: List[str] = []


def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global TESTS_PASSED, TESTS_FAILED, FAILURES
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                result = func(*args, **kwargs)
                print(f"✅ PASSED: {name}")
                TESTS_PASSED += 1
                return result
            except Exception as e:
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")
                traceback.print_exc()
                TESTS_FAILED += 1
                FAILURES.append(f"{name}: {e}")
                return None
        return wrapper
    return decorator


@test("Import Core Modules")
def test_imports():
    """Test that all modules can be imported."""
    from nokai import NokaiConfig, NokaiModel, NeuromorphicBrain
    from nokai.brain import BrainState
    from nokai.thalamus import ThalamusGateway
    from nokai.prefrontal import PrefrontalWorkingMemory
    from nokai.limbic import DopamineCircuit, StriatumSelector, MetacognitiveMonitor
    from nokai.memory import SemanticMemory, ConsolidationSystem
    from nokai.attention import AttentionController
    print("  All imports successful")
    return True


@test("Configuration Presets")
def test_config():
    """Test all configuration presets."""
    from nokai import NokaiConfig
    
    presets = ['nano', 'micro', 'mini', 'base', 'large']
    for preset in presets:
        config = getattr(NokaiConfig, preset)()
        params = config.estimate_parameters()
        vram = config.estimate_vram_mb()
        print(f"  {preset}: {params:,} params, {vram:.0f} MB VRAM")
    
    return True


@test("Tensor Dimension Flow")
def test_dimensions():
    """Verify tensor dimensions through the architecture."""
    from nokai import NokaiConfig
    from nokai.thalamus import ThalamusGateway
    from nokai.prefrontal import PrefrontalWorkingMemory
    from nokai.limbic import DopamineCircuit, StriatumSelector, MetacognitiveMonitor
    from nokai.memory import SemanticMemory
    from nokai.attention import AttentionController
    
    # Configuration
    dim = 128
    batch_size = 2
    seq_len = 32
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim)
    x_flat = x.mean(dim=1)  # [batch, dim]
    
    print(f"  Input shape: {x.shape}")
    
    # Test Thalamus
    thalamus = ThalamusGateway(input_dim=dim, num_clusters=16, sparsity_target=0.1)
    filtered, _ = thalamus(x)
    print(f"  Thalamus output: {filtered.shape}")
    assert filtered.shape[0] == batch_size
    assert filtered.shape[2] == dim
    
    # Test Working Memory
    wm = PrefrontalWorkingMemory(dim=dim, capacity=8)
    wm_out, _ = wm(x)
    print(f"  Working Memory output: {wm_out.shape}")
    assert wm_out.shape == x.shape
    
    # Test Dopamine Circuit
    dopamine = DopamineCircuit(state_dim=dim, hidden_dim=128)
    da_state, _ = dopamine(x_flat)
    print(f"  Dopamine level: {da_state.level:.3f}")
    assert 0 <= da_state.level <= 1
    
    # Test Striatum
    striatum = StriatumSelector(state_dim=dim, action_dim=dim, num_action_candidates=8)
    action, _ = striatum(x_flat, dopamine_level=0.5)
    print(f"  Striatum action shape: {action.shape}")
    assert action.shape == (batch_size, dim)
    
    # Test dACC
    dacc = MetacognitiveMonitor(state_dim=dim, hidden_dim=128)
    assessment, _ = dacc(x_flat)
    print(f"  dACC confidence: {assessment.confidence:.3f}")
    assert 0 <= assessment.confidence <= 1
    
    # Test Semantic Memory
    semantic = SemanticMemory(embedding_dim=dim, max_concepts=1000)
    sem_out, _ = semantic(x)
    print(f"  Semantic Memory output: {sem_out.shape}")
    assert sem_out.shape == x.shape
    
    # Test Attention Controller
    attention = AttentionController(state_dim=dim, num_modules=7)
    allocation, _ = attention(x_flat)
    print(f"  Attention allocation cortex: {allocation.cortex:.3f}")
    
    print("\n  All dimensions verified ✓")
    return True


@test("NeuromorphicBrain Initialization")
def test_brain_init():
    """Test brain initialization with nano config."""
    from nokai import NeuromorphicBrain, NokaiConfig
    
    config = NokaiConfig.nano()
    brain = NeuromorphicBrain(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in brain.parameters())
    print(f"  Parameters: {total_params:,}")
    
    return brain


@test("Forward Pass")
def test_forward(brain=None):
    """Test forward pass with random data."""
    if brain is None:
        from nokai import NeuromorphicBrain, NokaiConfig
        config = NokaiConfig.nano()
        brain = NeuromorphicBrain(config)
    
    batch_size = 2
    seq_len = 32
    
    # Random input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    outputs = brain(input_ids, return_brain_state=True)
    
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Brain state:")
    print(f"    - Dopamine: {outputs['brain_state'].dopamine_level:.3f}")
    print(f"    - Confidence: {outputs['brain_state'].confidence:.3f}")
    print(f"    - Cognitive Load: {outputs['brain_state'].cognitive_load:.3f}")
    
    expected_shape = (batch_size, seq_len, brain.config.vocab_size)
    assert outputs['logits'].shape == expected_shape, f"Expected {expected_shape}, got {outputs['logits'].shape}"
    
    return brain


@test("Backward Pass (Training)")
def test_backward(brain=None):
    """Test backward pass for training."""
    if brain is None:
        from nokai import NeuromorphicBrain, NokaiConfig
        config = NokaiConfig.nano()
        brain = NeuromorphicBrain(config)
    
    brain.train()
    
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward with labels
    outputs = brain(input_ids, labels=labels, store_memory=True)
    
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Backward
    outputs['loss'].backward()
    
    # Check gradients exist
    has_grads = False
    for name, param in brain.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break
    
    assert has_grads, "No gradients computed!"
    print("  Gradients computed successfully ✓")
    
    return brain


@test("Memory Systems")
def test_memory():
    """Test episodic and semantic memory."""
    from nokai import NeuromorphicBrain, NokaiConfig
    
    config = NokaiConfig.nano()
    brain = NeuromorphicBrain(config)
    brain.train()
    
    # Store some memories
    for i in range(5):
        input_ids = torch.randint(0, 1000, (1, 16))
        brain(input_ids, store_memory=True)
    
    # Check memory contents
    if brain.hippocampus is not None:
        print(f"  Hippocampus memories: {brain.hippocampus.memory_count}")
    
    semantic_concepts = brain.semantic_memory.num_concepts.item()
    print(f"  Semantic concepts: {semantic_concepts}")
    
    consolidation_queue = brain.consolidation.consolidation_queue.qsize()
    print(f"  Consolidation queue: {consolidation_queue}")
    
    return True


@test("Consolidation (Sleep)")
def test_consolidation():
    """Test memory consolidation."""
    from nokai import NeuromorphicBrain, NokaiConfig
    
    config = NokaiConfig.nano()
    brain = NeuromorphicBrain(config)
    brain.train()
    
    # Store some memories first
    for i in range(10):
        input_ids = torch.randint(0, 1000, (1, 16))
        brain(input_ids, store_memory=True)
    
    # Run consolidation
    stats = brain.consolidate(max_steps=50)
    
    print(f"  Consolidated: {stats.get('total_consolidated', 0)}")
    print(f"  Pruned: {stats.get('total_pruned', 0)}")
    
    return True


@test("Plasticity Stats")
def test_plasticity():
    """Test synaptic weight tracking."""
    from nokai import NeuromorphicBrain, NokaiConfig
    
    config = NokaiConfig.nano()
    brain = NeuromorphicBrain(config)
    
    stats = brain.get_plasticity_stats()
    print(f"  Plasticity stats: {stats}")
    
    for key, value in stats.items():
        assert isinstance(value, float), f"{key} should be float"
        print(f"    {key}: {value:.4f}")
    
    return True


@test("Energy Efficiency")
def test_energy():
    """Test energy/efficiency tracking."""
    from nokai import NeuromorphicBrain, NokaiConfig
    
    config = NokaiConfig.nano()
    brain = NeuromorphicBrain(config)
    
    # Run a few forward passes
    for _ in range(3):
        input_ids = torch.randint(0, 1000, (1, 16))
        brain(input_ids)
    
    stats = brain.get_energy_stats()
    print(f"  Energy stats: {stats}")
    
    return True


@test("Text Generation")
def test_generation():
    """Test text generation."""
    from nokai import NeuromorphicBrain, NokaiConfig
    
    config = NokaiConfig.nano()
    brain = NeuromorphicBrain(config)
    brain.eval()
    
    prompt = torch.randint(0, 1000, (1, 5))
    print(f"  Prompt shape: {prompt.shape}")
    
    generated = brain.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"  Generated shape: {generated.shape}")
    
    assert generated.shape[1] == 15, f"Expected 15 tokens, got {generated.shape[1]}"
    
    return True


@test("Legacy NokaiModel Compatibility")
def test_legacy_model():
    """Test that old NokaiModel still works."""
    from nokai import NokaiModel, NokaiConfig
    
    config = NokaiConfig.nano()
    model = NokaiModel(config)
    
    input_ids = torch.randint(0, 1000, (1, 16))
    outputs = model(input_ids)
    
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    global TESTS_PASSED, TESTS_FAILED, FAILURES
    
    print("\n" + "="*60)
    print("NŌKAI NEUROMORPHIC BRAIN - QA TEST SUITE")
    print("="*60)
    
    # Run tests in order
    test_imports()
    test_config()
    test_dimensions()
    brain = test_brain_init()
    test_forward(brain)
    test_backward()
    test_memory()
    test_consolidation()
    test_plasticity()
    test_energy()
    test_generation()
    test_legacy_model()
    
    # Report
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  ✅ Passed: {TESTS_PASSED}")
    print(f"  ❌ Failed: {TESTS_FAILED}")
    
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print(f"  - {f}")
    
    print("\n" + "="*60)
    
    if TESTS_FAILED == 0:
        print("🎉 SUCCÈS : Le cerveau est vivant!")
        print("="*60 + "\n")
        return True
    else:
        print("⚠️  ATTENTION : Certains tests ont échoué")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
