"""
Test script to verify Nōkai installation and basic functionality.
"""

import sys
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from nokai import NokaiConfig, NokaiModel
        print("  ✓ nokai.NokaiConfig")
        print("  ✓ nokai.NokaiModel")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    try:
        from nokai.cortex import CorticalColumn, CorticalLayer, Cortex
        print("  ✓ nokai.cortex")
    except ImportError as e:
        print(f"  ✗ Cortex import error: {e}")
        return False
    
    try:
        from nokai.hippocampus import HippocampalMemory
        print("  ✓ nokai.hippocampus")
    except ImportError as e:
        print(f"  ✗ Hippocampus import error: {e}")
        return False
    
    try:
        from nokai.oscillations import OscillatorNetwork
        print("  ✓ nokai.oscillations")
    except ImportError as e:
        print(f"  ✗ Oscillations import error: {e}")
        return False
    
    try:
        from nokai.learning import HebbianPlasticity, PredictiveCodingLayer
        print("  ✓ nokai.learning")
    except ImportError as e:
        print(f"  ✗ Learning import error: {e}")
        return False
    
    return True


def test_config():
    """Test configuration creation."""
    print("\nTesting configurations...")
    
    from nokai import NokaiConfig
    
    configs = ['nano', 'micro', 'mini', 'base', 'large']
    
    for name in configs:
        config = getattr(NokaiConfig, name)()
        params = config.estimate_parameters()
        vram = config.estimate_vram_mb()
        print(f"  {name:6s}: {params:>12,} params, {vram:>8.1f} MB VRAM")
    
    return True


def test_model_creation():
    """Test model instantiation."""
    print("\nTesting model creation...")
    
    from nokai import NokaiConfig, NokaiModel
    
    # Use nano for quick testing
    config = NokaiConfig.nano()
    config.device = "cpu"  # Use CPU for testing
    
    try:
        model = NokaiModel(config)
        print(f"  ✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Total parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass."""
    print("\nTesting forward pass...")
    
    from nokai import NokaiConfig, NokaiModel
    
    config = NokaiConfig.nano()
    config.device = "cpu"
    config.max_sequence_length = 32
    config.vocab_size = 1000
    
    model = NokaiModel(config)
    model.eval()
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs['logits']
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output shape: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    """Test text generation."""
    print("\nTesting generation...")
    
    from nokai import NokaiConfig, NokaiModel
    
    config = NokaiConfig.nano()
    config.device = "cpu"
    config.max_sequence_length = 64
    config.vocab_size = 1000
    
    model = NokaiModel(config)
    model.eval()
    
    try:
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
            )
        
        print(f"  ✓ Generation successful")
        print(f"  ✓ Generated {generated.shape[1] - 5} new tokens")
        
        return True
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cortical_column():
    """Test cortical column functionality."""
    print("\nTesting cortical column...")
    
    from nokai.config import CorticalColumnConfig
    from nokai.cortex.column import CorticalColumn
    
    config = CorticalColumnConfig(num_neurons=64, num_layers=3)
    column = CorticalColumn(config, column_id=0, input_dim=32)
    
    try:
        x = torch.randn(4, 32)
        output, metadata = column(x)
        
        print(f"  ✓ Column forward pass successful")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Mean sparsity: {metadata['mean_sparsity']:.2%}")
        
        # Test Hebbian updates
        updates = column.get_hebbian_update()
        column.apply_hebbian_update(updates)
        print(f"  ✓ Hebbian update applied")
        
        return True
    except Exception as e:
        print(f"  ✗ Cortical column test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_oscillations():
    """Test neural oscillations."""
    print("\nTesting oscillations...")
    
    from nokai.oscillations import OscillatorNetwork
    
    oscillators = OscillatorNetwork(num_oscillators=32)
    
    try:
        # Run a few steps
        for _ in range(10):
            oscillators.step(dt=0.001)
        
        coherence = oscillators.get_phase_coherence()
        modulation = oscillators.get_modulation()
        
        print(f"  ✓ Oscillation stepping successful")
        print(f"  ✓ Phase coherence: {coherence:.3f}")
        print(f"  ✓ Modulation shape: {modulation.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Oscillation test failed: {e}")
        return False


def test_hippocampus():
    """Test hippocampal memory."""
    print("\nTesting hippocampal memory...")
    
    from nokai.hippocampus import HippocampalMemory
    
    memory = HippocampalMemory(embedding_dim=64, memory_size=1000)
    
    try:
        # Store some memories
        keys = torch.randn(10, 64)
        values = torch.randn(10, 64)
        memory.store(keys, values)
        
        print(f"  ✓ Stored {memory.memory_count} memories")
        
        # Retrieve
        query = torch.randn(2, 64)
        output, metadata = memory(query)
        
        print(f"  ✓ Retrieved memories successfully")
        print(f"  ✓ Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Hippocampus test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Nōkai Architecture Verification")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        test_imports,
        test_config,
        test_cortical_column,
        test_oscillations,
        test_hippocampus,
        test_model_creation,
        test_forward_pass,
        test_generation,
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
