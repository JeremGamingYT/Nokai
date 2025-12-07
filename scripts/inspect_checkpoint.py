#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint Inspector - Analyze checkpoint architecture

Usage:
    python scripts/inspect_checkpoint.py checkpoints/brain_best.pt
"""

import sys
import torch
from pathlib import Path
from collections import defaultdict

def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint and show its architecture details."""
    
    print(f"\n{'='*60}")
    print(f"CHECKPOINT INSPECTOR")
    print(f"{'='*60}")
    print(f"File: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: File not found!")
        return
    
    # Load
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Total keys: {len(state_dict)}")
    
    # ============================================
    # INFER KEY DIMENSIONS
    # ============================================
    
    key_dims = {}
    
    # Embedding
    if 'embedding.weight' in state_dict:
        vocab_size, embedding_dim = state_dict['embedding.weight'].shape
        key_dims['vocab_size'] = vocab_size
        key_dims['embedding_dim'] = embedding_dim
        print(f"\n[Embedding]")
        print(f"  vocab_size: {vocab_size}")
        print(f"  embedding_dim: {embedding_dim}")
    
    # Position embedding
    if 'position_embedding.weight' in state_dict:
        max_seq, _ = state_dict['position_embedding.weight'].shape
        key_dims['max_sequence_length'] = max_seq
        print(f"  max_sequence_length: {max_seq}")
    
    # ============================================
    # DETECT ARCHITECTURE TYPE
    # ============================================
    
    has_dopamine = any('dopamine_circuit' in k for k in state_dict.keys())
    has_striatum = any('striatum' in k for k in state_dict.keys())
    has_dacc = any('dacc' in k for k in state_dict.keys())
    has_thalamus = any('thalamus' in k for k in state_dict.keys())
    has_working_memory = any('working_memory' in k for k in state_dict.keys())
    has_semantic = any('semantic_memory' in k for k in state_dict.keys())
    has_attention_controller = any('attention_controller' in k for k in state_dict.keys())
    has_cortex = any('cortex' in k for k in state_dict.keys())
    has_hippocampus = any('hippocampus' in k for k in state_dict.keys())
    
    is_neuromorphic = has_dopamine or has_striatum or has_dacc
    
    print(f"\n[Architecture Type]")
    if is_neuromorphic:
        print(f"  Type: NeuromorphicBrain")
    else:
        print(f"  Type: NokaiModel (legacy)")
    
    print(f"\n[Modules Detected]")
    modules = [
        ('Cortex', has_cortex),
        ('Thalamus', has_thalamus),
        ('Working Memory (PFC)', has_working_memory),
        ('Hippocampus', has_hippocampus),
        ('Semantic Memory', has_semantic),
        ('Dopamine Circuit', has_dopamine),
        ('Striatum', has_striatum),
        ('dACC', has_dacc),
        ('Attention Controller', has_attention_controller),
    ]
    for name, present in modules:
        symbol = "✓" if present else "✗"
        print(f"  {symbol} {name}")
    
    # ============================================
    # ANALYZE CORTEX STRUCTURE
    # ============================================
    
    # Count cortex layers and columns
    cortex_layers = set()
    cortex_columns = set()
    neuron_layers = set()
    
    for key in state_dict.keys():
        if 'cortex.layers' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i+1 < len(parts) and parts[i+1].isdigit():
                    cortex_layers.add(int(parts[i+1]))
                if part == 'columns' and i+1 < len(parts) and parts[i+1].isdigit():
                    cortex_columns.add(int(parts[i+1]))
    
    if cortex_layers:
        key_dims['num_cortex_layers'] = max(cortex_layers) + 1
        print(f"\n[Cortex Structure]")
        print(f"  num_layers: {max(cortex_layers) + 1}")
    if cortex_columns:
        key_dims['num_columns'] = max(cortex_columns) + 1
        print(f"  num_columns: {max(cortex_columns) + 1}")
    
    # ============================================
    # COUNT PARAMETERS
    # ============================================
    
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n[Parameters]")
    print(f"  Total: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1e6:.1f} MB (float32)")
    
    # Group by module
    module_params = defaultdict(int)
    for key, tensor in state_dict.items():
        module = key.split('.')[0]
        module_params[module] += tensor.numel()
    
    print(f"\n[Parameters by Module]")
    for module, params in sorted(module_params.items(), key=lambda x: -x[1]):
        pct = params / total_params * 100
        print(f"  {module}: {params:,} ({pct:.1f}%)")
    
    # ============================================
    # SUGGEST PRESET
    # ============================================
    
    print(f"\n[Suggested Preset]")
    embedding_dim = key_dims.get('embedding_dim', 0)
    
    if embedding_dim <= 128:
        suggested = 'nano'
    elif embedding_dim <= 192:
        suggested = 'micro'
    elif embedding_dim <= 256:
        suggested = 'mini'
    elif embedding_dim <= 512:
        suggested = 'base'
    else:
        suggested = 'large'
    
    print(f"  Based on embedding_dim={embedding_dim}: --preset {suggested}")
    
    # ============================================
    # SAMPLE KEYS
    # ============================================
    
    print(f"\n[Sample Keys (first 20)]")
    for key in list(state_dict.keys())[:20]:
        shape = tuple(state_dict[key].shape)
        print(f"  {key}: {shape}")
    
    print(f"\n{'='*60}")
    
    return key_dims


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_checkpoint.py <checkpoint_path>")
        print("Example: python scripts/inspect_checkpoint.py checkpoints/brain_best.pt")
        sys.exit(1)
    
    inspect_checkpoint(sys.argv[1])
