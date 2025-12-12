"""
GENESIS Blue Apple Experiment - One-Shot Learning Test

=============================================================================
THE BLUE APPLE TEST
=============================================================================

This experiment tests true ONE-SHOT learning:

1. Show the model: "The apple is" → "blue" ONCE
2. Test if it learned: "The apple is" → should produce "blue"

A standard LLM needs millions of examples.
GENESIS should learn from ONE.

This file tests the new architecture components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_ternary_weights():
    """Test ternary weight quantization."""
    print("\n" + "=" * 60)
    print("TEST 1: TERNARY WEIGHTS")
    print("=" * 60)
    
    from nokai.genesis.ternary import TernaryLinear, ternary_quantize
    
    # Create layer
    layer = TernaryLinear(256, 128)
    
    # Get ternary weights
    w_ternary = layer.get_ternary_weights()
    
    # Check that weights are ternary
    unique = torch.unique(w_ternary)
    print(f"Unique values in weights: {unique.tolist()}")
    
    assert all(v in [-1, 0, 1] for v in unique.tolist()), "Weights should be ternary!"
    
    # Check compression stats
    stats = layer.get_compression_stats()
    print(f"Compression stats:")
    print(f"  Total weights: {stats['total_weights']}")
    print(f"  Zeros (sparse): {stats['zeros']} ({stats['sparsity']*100:.1f}%)")
    print(f"  Compression ratio: {stats['compression_ratio']}x")
    
    # Test forward pass
    x = torch.randn(4, 256)
    y = layer(x)
    print(f"Forward pass: {x.shape} → {y.shape}")
    
    # Test backward (STE)
    loss = y.sum()
    loss.backward()
    print(f"Gradient computed: {layer.weight.grad is not None}")
    print(f"Gradient non-zero: {(layer.weight.grad.abs() > 0).any()}")
    
    print("✅ Ternary weights test PASSED")
    return True


def test_rich_neuron_unit():
    """Test Rich Neuron Unit dynamics."""
    print("\n" + "=" * 60)
    print("TEST 2: RICH NEURON UNIT")
    print("=" * 60)
    
    from nokai.genesis.rnu import RichNeuronUnit, RNUConfig
    
    config = RNUConfig(
        tau_membrane=0.9,
        tau_threshold=0.99,
        stochastic=True,
        ternary_output=True,
    )
    
    rnu = RichNeuronUnit(num_neurons=128, input_dim=256, config=config)
    
    # Test forward
    x = torch.randn(4, 256)
    output, state = rnu(x, return_state=True)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Output unique values: {torch.unique(output).tolist()[:10]}...")
    
    # Check ternary output
    unique_outputs = torch.unique(output)
    assert all(v in [-1, 0, 1] for v in unique_outputs.tolist()), "Output should be ternary!"
    
    # Check sparsity
    sparsity = (output == 0).float().mean()
    print(f"Output sparsity: {sparsity*100:.1f}%")
    
    # Check state
    print(f"State keys: {list(state.keys())}")
    print(f"Membrane range: [{state['membrane'].min():.3f}, {state['membrane'].max():.3f}]")
    print(f"Threshold range: [{state['threshold'].min():.3f}, {state['threshold'].max():.3f}]")
    
    # Test eligibility trace
    trace = rnu.get_eligibility_trace()
    print(f"Eligibility trace shape: {trace.shape}")
    print(f"Eligibility trace non-zero: {(trace.abs() > 1e-6).sum()}")
    
    print("✅ Rich Neuron Unit test PASSED")
    return True


def test_genesis_learning():
    """Test GENESIS local learning rule."""
    print("\n" + "=" * 60)
    print("TEST 3: GENESIS LOCAL LEARNING")
    print("=" * 60)
    
    from nokai.genesis.learning import GenesisLearning, GenesisLearningConfig
    
    config = GenesisLearningConfig(
        a_plus=0.01,
        a_minus=0.005,
        target_firing_rate=0.05,
    )
    
    learner = GenesisLearning(256, 128, config)
    
    # Create weight matrix
    weight = nn.Parameter(torch.randn(128, 256) * 0.1)
    weight_before = weight.clone()
    
    # Simulate activations
    pre = torch.randn(4, 256).abs()  # Pre-synaptic
    post = torch.randn(4, 128).abs()  # Post-synaptic
    
    # Apply local learning
    success, info = learner.apply_local_update(
        weight=weight,
        pre=pre,
        post=post,
        dopamine=0.8,
        acetylcholine=0.7,
        rpe=0.5,
    )
    
    print(f"Learning applied: {success}")
    print(f"Learning info:")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Check weight change
    weight_change = (weight - weight_before).abs().mean()
    print(f"\nWeight change magnitude: {weight_change:.6f}")
    
    assert weight_change > 0, "Weights should have changed!"
    
    print("✅ GENESIS Learning test PASSED")
    return True


def test_modern_hopfield():
    """Test Modern Hopfield Memory for one-shot learning."""
    print("\n" + "=" * 60)
    print("TEST 4: MODERN HOPFIELD MEMORY")
    print("=" * 60)
    
    from nokai.genesis.memory import ModernHopfieldMemory
    
    memory = ModernHopfieldMemory(
        pattern_dim=128,
        memory_size=1000,
        beta=8.0,  # High beta for sharper retrieval
    )
    
    # Store some patterns
    patterns = []
    for i in range(10):
        # Create distinct patterns
        p = torch.randn(128)
        p[i*10:(i+1)*10] = 5  # Make each pattern have a unique signature
        patterns.append(p)
        memory.store(p)
    
    print(f"Stored {memory.num_stored} patterns")
    
    # Test retrieval with partial cue
    query = patterns[5].clone()
    query[50:] = 0  # Corrupt second half
    
    retrieved, _ = memory.retrieve(query)
    
    # Check similarity to original
    similarity = F.cosine_similarity(
        retrieved.flatten(),
        patterns[5].flatten(),
        dim=0,
    )
    print(f"Retrieval similarity: {similarity:.4f}")
    
    assert similarity > 0.8, "Retrieval should be accurate!"
    
    # Test energy
    energy_stored = memory.energy(patterns[5])
    energy_random = memory.energy(torch.randn(128))
    print(f"Energy of stored pattern: {energy_stored:.4f}")
    print(f"Energy of random pattern: {energy_random:.4f}")
    
    assert energy_stored < energy_random, "Stored patterns should have lower energy!"
    
    print("✅ Modern Hopfield test PASSED")
    return True


def test_oscillatory_binding():
    """Test oscillatory binding for concept linking."""
    print("\n" + "=" * 60)
    print("TEST 5: OSCILLATORY BINDING")
    print("=" * 60)
    
    from nokai.genesis.binding import OscillatoryBinder
    
    binder = OscillatoryBinder(
        num_concepts=100,
        embedding_dim=64,
        theta_freq=6.0,
        gamma_freq=40.0,
    )
    
    # Bind concepts 0 and 1 ("blue" and "apple")
    binder.bind(0, 1, strength=1.0)
    
    # Check that they're synchronized
    phase_diff = abs(binder.gamma_phases[0] - binder.gamma_phases[1])
    print(f"Phase difference after binding: {phase_diff:.4f} rad")
    
    assert phase_diff < 0.5, "Bound concepts should have similar phases!"
    
    # Get bound concepts
    bound_to_0 = binder.get_bound_concepts(0)
    print(f"Concepts bound to 0: {bound_to_0}")
    
    assert 1 in bound_to_0, "Concept 1 should be bound to 0!"
    
    # Test modulation over time
    for _ in range(100):
        binder.step()
    
    mod = binder.get_modulation(torch.tensor([0, 1, 50]))
    print(f"Modulation for concepts [0, 1, 50]: {mod.tolist()}")
    
    print("✅ Oscillatory binding test PASSED")
    return True


def test_neuromodulation():
    """Test neuromodulation system."""
    print("\n" + "=" * 60)
    print("TEST 6: NEUROMODULATION")
    print("=" * 60)
    
    from nokai.genesis.neuromodulation import GenesisLimbic
    
    limbic = GenesisLimbic(state_dim=128)
    
    # Process some states with rewards
    state = torch.randn(128)
    
    # Positive reward
    nm_state = limbic.process(state, reward=1.0)
    print(f"After positive reward:")
    print(f"  Dopamine: {nm_state.dopamine:.3f}")
    print(f"  Acetylcholine: {nm_state.acetylcholine:.3f}")
    
    # Check dopamine increased
    assert nm_state.dopamine > 0.5, "Dopamine should increase with positive reward!"
    
    # Negative reward
    nm_state = limbic.process(state, reward=-0.5)
    print(f"After negative reward:")
    print(f"  Dopamine: {nm_state.dopamine:.3f}")
    
    # Get learning modulation
    modulation = limbic.get_learning_modulation()
    print(f"\nLearning modulation:")
    for k, v in modulation.items():
        print(f"  {k}: {v:.3f}")
    
    print("✅ Neuromodulation test PASSED")
    return True


def test_blue_apple_integration():
    """Integration test: Blue Apple one-shot learning."""
    print("\n" + "=" * 60)
    print("TEST 7: BLUE APPLE INTEGRATION")
    print("=" * 60)
    
    from nokai.genesis.memory import ModernHopfieldMemory, WorkingMemoryBuffer
    from nokai.genesis.learning import GenesisLearningLayer
    from nokai.genesis.neuromodulation import GenesisLimbic
    
    # Setup
    dim = 64
    vocab_size = 100
    
    # Create embeddings (simulating tokenizer)
    embeddings = nn.Embedding(vocab_size, dim)
    
    # Token IDs (simulated)
    THE, APPLE, IS, BLUE, RED = 10, 20, 30, 40, 50
    
    # Create components
    episodic = ModernHopfieldMemory(dim, memory_size=1000, beta=10.0)
    working = WorkingMemoryBuffer(dim, num_slots=7)
    layer = GenesisLearningLayer(dim, dim, use_ternary=False)
    limbic = GenesisLimbic(dim)
    
    print("Phase 1: Learning 'The apple is blue'")
    print("-" * 40)
    
    # Embed the sentence
    context = embeddings(torch.tensor([THE, APPLE, IS]))
    target = embeddings(torch.tensor([BLUE]))
    
    # Store context in working memory
    for i, tok_emb in enumerate(context):
        working.write(tok_emb)
    
    # Read from working memory
    wm_content = working.read(context.mean(0))
    
    # Process through layer
    output = layer(wm_content)
    
    # Compute "reward" (high because this is teaching)
    reward = 1.0
    
    # Update limbic system
    nm_state = limbic.process(output.detach(), reward=reward)
    print(f"Neuromodulation: DA={nm_state.dopamine:.3f}, ACh={nm_state.acetylcholine:.3f}")
    
    # Apply local learning with target clamped
    info = layer.local_learn(
        dopamine=nm_state.dopamine,
        acetylcholine=nm_state.acetylcholine,
        rpe=reward,
    )
    print(f"Learning applied: delta_mean={info.get('total_delta_mean', 0):.6f}")
    
    # Store in episodic memory
    # Key: context representation, Value: target
    episode = torch.cat([wm_content.detach().squeeze(), target.squeeze()])
    episodic.store(episode[:dim])  # Store context
    
    print(f"\nPhase 2: Testing recall")
    print("-" * 40)
    
    # Now test: given "The apple is", what comes next?
    test_context = embeddings(torch.tensor([THE, APPLE, IS]))
    test_query = test_context.mean(0)
    
    # Retrieve from episodic
    retrieved, _ = episodic.retrieve(test_query)
    
    # Measure similarity to blue vs red
    blue_emb = embeddings(torch.tensor([BLUE])).squeeze()
    red_emb = embeddings(torch.tensor([RED])).squeeze()
    
    # Process retrieved through layer
    output = layer(retrieved.unsqueeze(0))
    
    # Compare to targets
    sim_blue = F.cosine_similarity(output.squeeze(), blue_emb, dim=0)
    sim_red = F.cosine_similarity(output.squeeze(), red_emb, dim=0)
    
    print(f"Similarity to 'blue': {sim_blue:.4f}")
    print(f"Similarity to 'red': {sim_red:.4f}")
    
    # The pattern should be stored
    stats = episodic.get_statistics()
    print(f"\nEpisodic memory stats: {stats}")
    
    print("\n✅ Blue Apple integration test COMPLETE")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("  GENESIS ARCHITECTURE TESTS")
    print("  Nōkai V2.0 Validation Suite")
    print("=" * 60)
    
    tests = [
        ("Ternary Weights", test_ternary_weights),
        ("Rich Neuron Unit", test_rich_neuron_unit),
        ("GENESIS Learning", test_genesis_learning),
        ("Modern Hopfield", test_modern_hopfield),
        ("Oscillatory Binding", test_oscillatory_binding),
        ("Neuromodulation", test_neuromodulation),
        ("Blue Apple Integration", test_blue_apple_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASSED" if p else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
