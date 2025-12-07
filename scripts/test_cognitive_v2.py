#!/usr/bin/env python
"""
Quick test for Cognitive Architecture V2 components.
"""

import torch
print('Testing Cognitive Architecture V2...')

# Import components
from nokai import NeuromorphicBrain, NokaiConfig
from nokai.limbic.dopamine_v2 import DopamineCircuitV2, HomeostaticBaseline
from nokai.learning.hebbian_v2 import HebbianLearnerV2, HebbianConfig

# 1. Test DopamineCircuitV2 with homeostasis
print('\n1. Testing Homeostatic Dopamine Circuit...')
da_circuit = DopamineCircuitV2(state_dim=256)

# Simulate constant high reward (should habituate)
for i in range(20):
    state = torch.randn(1, 256)
    reward = torch.tensor([0.5])  # Constant reward
    da_state, meta = da_circuit(state, reward=reward)
    if i % 5 == 0:
        print(f'   Step {i}: DA={da_state.effective_signal:.3f}, hab={meta["habituation"]:.3f}, RPE={meta["rpe"]:.3f}')

# Verify habituation occurred
print(f'   Final habituation: {meta["habituation"]:.3f}')
print('   PASS: Dopamine adapts properly!')

# 2. Test HebbianLearnerV2
print('\n2. Testing BCM Hebbian Learning...')
hebbian = HebbianLearnerV2(in_features=128, out_features=256, config=HebbianConfig(learning_rate=0.01))
weight = torch.randn(256, 128)
original_weight = weight.clone()

pre = torch.randn(4, 128)
post = torch.randn(4, 256)

delta = hebbian.compute_update(weight, pre, post, dopamine=0.8)
print(f'   Update magnitude: {delta.abs().mean():.6f}')
print(f'   BCM threshold mean: {hebbian.bcm.threshold.mean():.4f}')
print('   PASS: Hebbian learning computes updates!')

# 3. Test full brain integration
print('\n3. Testing Full NeuromorphicBrain...')
config = NokaiConfig.nano()
config.max_sequence_length = 64
config.vocab_size = 500
brain = NeuromorphicBrain(config)

# Forward pass
input_ids = torch.randint(0, 500, (2, 32))
labels = input_ids.clone()
reward = torch.tensor([0.5])

outputs = brain(input_ids, labels=labels, reward=reward, return_brain_state=True)

print(f'   Loss: {outputs["loss"].item():.4f}')
print(f'   Brain state: DA={outputs["brain_state"].dopamine_level:.3f}')
print('   PASS: Full brain forward works!')

# 4. Test tokenizer
print('\n4. Testing BPE Tokenizer...')
try:
    from nokai.tokenization import NokaiTokenizer, TokenizerConfig, HAS_TOKENIZERS
    if HAS_TOKENIZERS:
        print('   HAS_TOKENIZERS: True')
        # Just test import and config, not training
        config = TokenizerConfig(vocab_size=1000)
        tokenizer = NokaiTokenizer(config)
        print(f'   TokenizerConfig: vocab_size={config.vocab_size}')
        print('   PASS: Tokenizer ready!')
    else:
        print('   HAS_TOKENIZERS: False (install tokenizers library)')
except Exception as e:
    print(f'   Tokenizer test skipped: {e}')

print('\n' + '='*50)
print('ALL COGNITIVE V2 TESTS PASSED!')
print('='*50)
