#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nōkai Neuromorphic Brain - Inference & Testing Script

Test your trained model with:
    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/brain_best.pt
    python scripts/chat.py --mode interactive
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig


# ============================================
# SIMPLE TOKENIZER (same as training)
# ============================================

class SimpleTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.char_to_id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.id_to_char = {0: '<pad>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.next_id = 4
    
    def encode(self, text: str) -> list:
        ids = [self.bos_token_id]
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        return ids
    
    def decode(self, ids: list) -> str:
        chars = []
        for id in ids:
            if id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            chars.append(self.id_to_char.get(id, ''))
        return ''.join(chars)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(data['vocab_size'])
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        return tokenizer


# ============================================
# INFERENCE
# ============================================

def load_model(checkpoint_path: str, preset: str = "mini", device: str = "cuda"):
    """
    Load trained model with auto-detection of architecture and config.
    
    The script tries to:
    1. Load config.json from checkpoints/ to get the correct preset
    2. Detect if it's NeuromorphicBrain or legacy NokaiModel
    3. Handle dimension mismatches gracefully
    """
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Try to load config.json for auto-detection
    checkpoint_dir = Path(checkpoint_path).parent
    config_json_path = checkpoint_dir / "config.json"
    
    detected_preset = None
    if config_json_path.exists():
        try:
            with open(config_json_path, 'r') as f:
                saved_config = json.load(f)
            detected_preset = saved_config.get('preset')
            if detected_preset:
                print(f"Auto-detected preset from config.json: {detected_preset}")
                if detected_preset != preset:
                    print(f"  (Overriding --preset {preset} with saved {detected_preset})")
                preset = detected_preset
        except Exception as e:
            print(f"Could not load config.json: {e}")
    
    # Config
    config_methods = {
        "nano": NokaiConfig.nano,
        "micro": NokaiConfig.micro,
        "mini": NokaiConfig.mini,
        "base": NokaiConfig.base,
        "large": NokaiConfig.large,
    }
    config = config_methods[preset]()
    
    # Check if checkpoint exists
    checkpoint_exists = Path(checkpoint_path).exists()
    if not checkpoint_exists:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Using random initialization...")
        brain = NeuromorphicBrain(config)
        brain = brain.to(device)
        brain.eval()
        return brain, config
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Detect architecture type
    is_neuromorphic = any('dopamine_circuit' in k or 'striatum' in k or 'dacc' in k 
                         for k in state_dict.keys())
    is_legacy = any('cortex.layers' in k for k in state_dict.keys()) and not is_neuromorphic
    
    # Try to infer embedding_dim from checkpoint
    for key in state_dict.keys():
        if 'embedding.weight' in key:
            _, checkpoint_dim = state_dict[key].shape
            if checkpoint_dim != config.embedding_dim:
                print(f"\nWARNING: Dimension mismatch detected!")
                print(f"  Checkpoint embedding_dim: {checkpoint_dim}")
                print(f"  Config embedding_dim: {config.embedding_dim}")
                
                # Try to find matching preset
                dim_to_preset = {64: 'nano', 128: 'nano', 256: 'mini', 512: 'base', 1024: 'large'}
                suggested = dim_to_preset.get(checkpoint_dim)
                if suggested:
                    print(f"  Suggested fix: use --preset {suggested}")
                    if detected_preset is None:  # Only auto-fix if no config.json
                        print(f"  Auto-switching to preset: {suggested}")
                        config = config_methods[suggested]()
                        preset = suggested
            break
    
    # Load based on architecture type
    if is_legacy:
        print("Detected LEGACY NokaiModel checkpoint")
        print("Loading with NokaiModel instead of NeuromorphicBrain...")
        
        from nokai import NokaiModel
        model = NokaiModel(config)
        try:
            model.load_state_dict(state_dict)
            print("Legacy model loaded successfully!")
        except Exception as e:
            print(f"Failed to load legacy model: {e}")
            print("The checkpoint may be incompatible.")
            return None, config
        
        model = model.to(device)
        model.eval()
        return model, config
    
    # Create NeuromorphicBrain
    print("Loading NeuromorphicBrain...")
    brain = NeuromorphicBrain(config)
    
    # Try loading weights
    try:
        brain.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully!")
    except RuntimeError as e:
        error_msg = str(e)
        
        if "size mismatch" in error_msg:
            print("\nERROR: Dimension mismatch between checkpoint and model!")
            print("This usually means you trained with a different --preset.")
            print("\nTry one of these commands:")
            print("  python scripts/chat.py --preset nano --checkpoint ...")
            print("  python scripts/chat.py --preset micro --checkpoint ...")
            print("  python scripts/chat.py --preset mini --checkpoint ...")
            print("\nOr retrain with the correct preset to match your target.")
            print("\nUsing random initialization instead...")
        else:
            print(f"Strict loading failed: {error_msg[:300]}...")
            print("Trying flexible loading (strict=False)...")
            
            try:
                missing, unexpected = brain.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"  Missing keys: {len(missing)} (random init)")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)} (ignored)")
                print("Partial weights loaded!")
            except Exception as e2:
                print(f"Flexible loading also failed: {e2}")
                print("Using random initialization...")
    
    brain = brain.to(device)
    brain.eval()
    
    return brain, config


def generate_text(
    brain: NeuromorphicBrain,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
    show_brain_state: bool = False,
) -> str:
    """Generate text from prompt."""
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nGenerating (temperature={temperature}, top_k={top_k}, top_p={top_p})...")
    
    # Generate
    with torch.no_grad():
        output_ids = brain.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Decode
    generated = tokenizer.decode(output_ids[0].tolist())
    
    # Show brain state if requested
    if show_brain_state:
        outputs = brain(output_ids, return_brain_state=True)
        state = outputs['brain_state']
        
        print(f"\nBrain State:")
        print(f"  Dopamine: {state.dopamine_level:.3f}")
        print(f"  Confidence: {state.confidence:.3f}")
        print(f"  Cognitive Load: {state.cognitive_load:.3f}")
        print(f"  Oscillation Phase: {state.oscillation_phase:.3f}")
    
    return generated


def interactive_mode(brain, tokenizer, device, args):
    """Interactive chat mode."""
    
    print("\n" + "="*60)
    print("NOKAI INTERACTIVE MODE")
    print("="*60)
    print("Commands:")
    print("  /quit       - Exit")
    print("  /state      - Show brain state")
    print("  /consolidate- Run memory consolidation")
    print("  /plasticity - Show plasticity stats")
    print("  /temp X     - Set temperature (0.1-2.0)")
    print("="*60 + "\n")
    
    temperature = args.temperature
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            
            if not prompt:
                continue
            
            # Commands
            if prompt.lower() == "/quit":
                print("Goodbye!")
                break
            
            elif prompt.lower() == "/state":
                dummy = torch.randint(0, 1000, (1, 10), device=device)
                outputs = brain(dummy, return_brain_state=True)
                state = outputs['brain_state']
                print(f"\nBrain State:")
                print(f"  Dopamine: {state.dopamine_level:.3f}")
                print(f"  Confidence: {state.confidence:.3f}")
                print(f"  Cognitive Load: {state.cognitive_load:.3f}")
                continue
            
            elif prompt.lower() == "/consolidate":
                print("\nRunning memory consolidation...")
                brain.train()
                stats = brain.consolidate(max_steps=50)
                brain.eval()
                print(f"  Consolidated: {stats.get('total_consolidated', 0)}")
                print(f"  Pruned: {stats.get('total_pruned', 0)}")
                continue
            
            elif prompt.lower() == "/plasticity":
                stats = brain.get_plasticity_stats()
                print(f"\nPlasticity Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v:.4f}")
                continue
            
            elif prompt.lower().startswith("/temp "):
                try:
                    temperature = float(prompt.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"Temperature set to: {temperature}")
                except:
                    print("Usage: /temp 0.8")
                continue
            
            # Generate response
            generated = generate_text(
                brain, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                show_brain_state=args.show_state,
            )
            
            print(f"\nNokai: {generated}")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


def benchmark_mode(brain, tokenizer, device, args):
    """Benchmark inference speed."""
    
    import time
    
    print("\n" + "="*60)
    print("BENCHMARK MODE")
    print("="*60)
    
    # Warmup
    print("Warming up...")
    dummy = torch.randint(0, 1000, (1, 32), device=device)
    for _ in range(3):
        with torch.no_grad():
            brain(dummy)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    print("\nBenchmarking forward pass...")
    times = []
    for seq_len in [32, 64, 128, 256, 512]:
        dummy = torch.randint(0, 1000, (1, seq_len), device=device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                brain(dummy)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / 10
        tokens_per_sec = seq_len / elapsed
        
        print(f"  Seq len {seq_len}: {elapsed*1000:.1f}ms ({tokens_per_sec:.0f} tok/s)")
    
    # Benchmark generation
    print("\nBenchmarking generation...")
    prompt = torch.randint(0, 1000, (1, 10), device=device)
    
    for new_tokens in [20, 50, 100]:
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            brain.generate(prompt, max_new_tokens=new_tokens)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        tokens_per_sec = new_tokens / elapsed
        
        print(f"  Generate {new_tokens} tokens: {elapsed:.2f}s ({tokens_per_sec:.0f} tok/s)")
    
    print("\n" + "="*60)


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Nokai Inference & Testing")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default="checkpoints/brain_best.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json",
                       help="Path to tokenizer")
    parser.add_argument("--preset", type=str, default="mini",
                       choices=["nano", "micro", "mini", "base", "large"])
    
    # Mode
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "generate", "benchmark"],
                       help="Run mode")
    
    # Generation
    parser.add_argument("--prompt", type=str, default="Hello, I am",
                       help="Prompt for generate mode")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # Options
    parser.add_argument("--show_state", action="store_true",
                       help="Show brain state after generation")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")
    
    # Load model
    brain, config = load_model(args.checkpoint, args.preset, device)
    
    # Load tokenizer
    if Path(args.tokenizer).exists():
        tokenizer = SimpleTokenizer.load(args.tokenizer)
        print(f"Tokenizer loaded (vocab: {len(tokenizer.char_to_id)})")
    else:
        print("No tokenizer found, using default")
        tokenizer = SimpleTokenizer(config.vocab_size)
    
    # Run mode
    if args.mode == "interactive":
        interactive_mode(brain, tokenizer, device, args)
    
    elif args.mode == "generate":
        generated = generate_text(
            brain, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            show_brain_state=args.show_state,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}")
    
    elif args.mode == "benchmark":
        benchmark_mode(brain, tokenizer, device, args)


if __name__ == "__main__":
    main()
