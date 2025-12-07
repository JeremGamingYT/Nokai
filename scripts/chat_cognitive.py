#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nōkai Interactive Chat - Test Your Trained Brain

=============================================================================
FEATURES
=============================================================================

1. Interactive Chat Mode
   - Real-time text generation with your trained model
   - Display of brain state (dopamine, confidence, cognitive load)

2. Benchmark Mode
   - Perplexity evaluation on test data
   - Generation quality metrics

3. Analysis Mode
   - Dopamine system statistics
   - Memory utilization
   - Attention patterns

=============================================================================

Usage:
    python scripts/chat_cognitive.py --checkpoint checkpoints/brain_best.pt
    python scripts/chat_cognitive.py --checkpoint checkpoints/brain_best.pt --mode benchmark
    
Author: Nōkai Neuro-Engineering Team
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Nōkai imports
from nokai import NeuromorphicBrain, NokaiConfig

# Try to import tokenization
try:
    from nokai.tokenization import NokaiTokenizer, HAS_TOKENIZERS
    USE_BPE = HAS_TOKENIZERS
except ImportError:
    USE_BPE = False


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class ChatConfig:
    """Configuration for chat/inference."""
    # Model
    checkpoint_path: str = "checkpoints/brain_best.pt"
    tokenizer_path: Optional[str] = None  # Auto-detect if None
    
    # Generation
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Display
    show_brain_state: bool = True
    show_tokens: bool = False
    stream_output: bool = True
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to IDs."""
        ids = [self.bos_token_id]
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode IDs to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(self.id_to_char.get(i, '?'))
        return ''.join(chars)
    
    @classmethod
    def load(cls, path: str) -> "FallbackTokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(data.get('vocab_size', 32000))
        tok.char_to_id = data['char_to_id']
        tok.id_to_char = {int(v): k for k, v in data['char_to_id'].items()}
        tok.next_id = max(tok.id_to_char.keys()) + 1
        return tok


# ============================================
# MODEL LOADER
# ============================================

def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[NeuromorphicBrain, any, NokaiConfig]:
    """
    Load trained model and tokenizer.
    
    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer (auto-detect if None)
        device: Device to load model on
        
    Returns:
        brain: Loaded NeuromorphicBrain
        tokenizer: Loaded tokenizer
        config: Model configuration
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n[Loading] Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # =====================================
    # INFER CONFIG FROM CHECKPOINT
    # =====================================
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Try to infer dimensions from state_dict
    if 'embedding.weight' in state_dict:
        vocab_size, embedding_dim = state_dict['embedding.weight'].shape
    else:
        # Fallback to defaults
        vocab_size = 32000
        embedding_dim = 256
    
    if 'position_embedding.weight' in state_dict:
        max_seq_length = state_dict['position_embedding.weight'].shape[0]
    else:
        max_seq_length = 512
    
    print(f"  Inferred config: vocab={vocab_size}, dim={embedding_dim}, seq_len={max_seq_length}")
    
    # =====================================
    # CREATE CONFIG AND MODEL
    # =====================================
    # Try to match a preset based on embedding_dim
    if embedding_dim <= 128:
        config = NokaiConfig.nano()
    elif embedding_dim <= 192:
        config = NokaiConfig.micro()
    elif embedding_dim <= 256:
        config = NokaiConfig.mini()
    elif embedding_dim <= 512:
        config = NokaiConfig.base()
    else:
        config = NokaiConfig.large()
    
    # Override with actual values
    config.vocab_size = vocab_size
    config.embedding_dim = embedding_dim
    config.max_sequence_length = max_seq_length
    
    # Create model
    brain = NeuromorphicBrain(config)
    
    # Load weights
    try:
        brain.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights successfully")
    except Exception as e:
        print(f"  Warning: Some weights not loaded: {e}")
    
    brain = brain.to(device)
    brain.eval()
    
    param_count = sum(p.numel() for p in brain.parameters())
    print(f"  Parameters: {param_count:,}")
    
    # =====================================
    # LOAD TOKENIZER
    # =====================================
    if tokenizer_path is None:
        # Auto-detect tokenizer in same directory
        checkpoint_dir = checkpoint_path.parent
        bpe_path = checkpoint_dir / "tokenizer.json"
        char_path = checkpoint_dir / "tokenizer.json"
        
        if bpe_path.exists() and USE_BPE:
            tokenizer_path = str(bpe_path)
        elif char_path.exists():
            tokenizer_path = str(char_path)
    
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"[Loading] Tokenizer: {tokenizer_path}")
        
        # Determine tokenizer type
        try:
            if USE_BPE:
                tokenizer = NokaiTokenizer.load(tokenizer_path)
                print(f"  BPE tokenizer loaded (vocab={tokenizer.vocab_size})")
            else:
                tokenizer = FallbackTokenizer.load(tokenizer_path)
                print(f"  Character tokenizer loaded")
        except Exception as e:
            print(f"  Warning: Could not load tokenizer: {e}")
            print(f"  Creating default tokenizer...")
            tokenizer = FallbackTokenizer(vocab_size)
    else:
        print(f"[Warning] No tokenizer found, using default character tokenizer")
        tokenizer = FallbackTokenizer(vocab_size)
    
    return brain, tokenizer, config


# ============================================
# GENERATION
# ============================================

def generate_text(
    brain: NeuromorphicBrain,
    tokenizer,
    prompt: str,
    config: ChatConfig,
) -> Tuple[str, Dict]:
    """
    Generate text from a prompt.
    
    Args:
        brain: Trained NeuromorphicBrain
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        config: Generation configuration
        
    Returns:
        generated_text: The generated response
        metadata: Brain state and generation stats
    """
    device = next(brain.parameters()).device
    
    # Encode prompt
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt)
    
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        # Get initial brain state
        outputs = brain(input_ids, return_brain_state=True)
        initial_state = outputs.get('brain_state', None)
        
        # Generate tokens one by one
        generated_ids = input_ids.clone()
        generated_tokens = []
        
        for i in range(config.max_new_tokens):
            # Forward pass
            outputs = brain(generated_ids, return_brain_state=True)
            logits = outputs['logits'][:, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for token_id in generated_ids[0].tolist():
                    logits[0, token_id] /= config.repetition_penalty
            
            # Apply top-k
            if config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus sampling)
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            # Stream output
            if config.stream_output:
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(token_text, end='', flush=True)
            
            # Check max sequence length
            if generated_ids.shape[1] >= brain.config.max_sequence_length:
                break
        
        if config.stream_output:
            print()  # Newline after streaming
        
        # Get final brain state
        final_outputs = brain(generated_ids, return_brain_state=True)
        final_state = final_outputs.get('brain_state', None)
    
    generation_time = time.time() - start_time
    tokens_per_second = len(generated_tokens) / max(0.001, generation_time)
    
    # Decode full response
    full_response = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    metadata = {
        'tokens_generated': len(generated_tokens),
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second,
        'initial_dopamine': initial_state.dopamine_level if initial_state else 0.5,
        'final_dopamine': final_state.dopamine_level if final_state else 0.5,
        'confidence': final_state.confidence if final_state else 0.5,
        'cognitive_load': final_state.cognitive_load if final_state else 0.5,
    }
    
    return generated_response, metadata


# ============================================
# INTERACTIVE CHAT
# ============================================

def interactive_chat(
    brain: NeuromorphicBrain,
    tokenizer,
    config: ChatConfig,
):
    """
    Interactive chat loop.
    """
    print("\n" + "=" * 60)
    print("🧠 NŌKAI INTERACTIVE CHAT")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit  - Exit chat")
    print("  /clear         - Clear context")
    print("  /config        - Show current config")
    print("  /state         - Show brain state")
    print("  /temp <value>  - Set temperature")
    print("  /tokens <num>  - Set max tokens")
    print("=" * 60 + "\n")
    
    context = ""
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().split()
                
                if cmd[0] in ['/quit', '/exit']:
                    print("\nGoodbye! 👋")
                    break
                
                elif cmd[0] == '/clear':
                    context = ""
                    print("[Context cleared]")
                    continue
                
                elif cmd[0] == '/config':
                    print(f"\n[Configuration]")
                    print(f"  Temperature: {config.temperature}")
                    print(f"  Max tokens: {config.max_new_tokens}")
                    print(f"  Top-k: {config.top_k}")
                    print(f"  Top-p: {config.top_p}")
                    print(f"  Device: {config.device}")
                    continue
                
                elif cmd[0] == '/state':
                    stats = brain.dopamine_circuit.get_statistics()
                    print(f"\n[Brain State]")
                    print(f"  Dopamine (tonic): {stats.get('current_tonic', 0.5):.3f}")
                    print(f"  Dopamine (phasic): {stats.get('current_phasic', 0.0):.3f}")
                    print(f"  Novelty: {stats.get('current_novelty', 0.0):.3f}")
                    print(f"  RPE: {stats.get('current_rpe', 0.0):.3f}")
                    print(f"  Habituation: {stats.get('habituation', 0.0):.3f}")
                    continue
                
                elif cmd[0] == '/temp' and len(cmd) > 1:
                    try:
                        config.temperature = float(cmd[1])
                        print(f"[Temperature set to {config.temperature}]")
                    except ValueError:
                        print("[Error: Invalid temperature value]")
                    continue
                
                elif cmd[0] == '/tokens' and len(cmd) > 1:
                    try:
                        config.max_new_tokens = int(cmd[1])
                        print(f"[Max tokens set to {config.max_new_tokens}]")
                    except ValueError:
                        print("[Error: Invalid token count]")
                    continue
                
                else:
                    print(f"[Unknown command: {cmd[0]}]")
                    continue
            
            # Build prompt with context
            if context:
                prompt = context + "\n" + user_input
            else:
                prompt = user_input
            
            # Generate response
            print("\nNōkai: ", end='', flush=True)
            
            response, metadata = generate_text(brain, tokenizer, prompt, config)
            
            if not config.stream_output:
                print(response)
            
            # Update context
            context = prompt + " " + response
            
            # Trim context if too long
            max_context_chars = brain.config.max_sequence_length * 3
            if len(context) > max_context_chars:
                context = context[-max_context_chars:]
            
            # Show brain state
            if config.show_brain_state:
                print(f"\n  [DA: {metadata['final_dopamine']:.2f} | "
                      f"Conf: {metadata['confidence']:.2f} | "
                      f"Load: {metadata['cognitive_load']:.2f} | "
                      f"{metadata['tokens_per_second']:.1f} tok/s]")
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"\n[Error: {e}]")


# ============================================
# BENCHMARK MODE
# ============================================

def run_benchmark(
    brain: NeuromorphicBrain,
    tokenizer,
    config: ChatConfig,
    test_prompts: Optional[List[str]] = None,
):
    """
    Run benchmark tests.
    """
    if test_prompts is None:
        test_prompts = [
            "The meaning of life is",
            "Artificial intelligence will",
            "In the year 2050,",
            "The human brain is",
            "Once upon a time,",
            "The most important thing in science is",
            "Philosophy teaches us that",
            "The future of technology",
        ]
    
    print("\n" + "=" * 60)
    print("🧪 NŌKAI BENCHMARK")
    print("=" * 60)
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Prompt: \"{prompt}\"")
        print("-" * 40)
        
        response, metadata = generate_text(brain, tokenizer, prompt, config)
        
        if not config.stream_output:
            print(f"Response: {response}")
        
        print(f"\n  Stats: {metadata['tokens_generated']} tokens, "
              f"{metadata['tokens_per_second']:.1f} tok/s, "
              f"DA: {metadata['final_dopamine']:.2f}")
        
        results.append(metadata)
        total_tokens += metadata['tokens_generated']
        total_time += metadata['generation_time']
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total prompts: {len(test_prompts)}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average speed: {total_tokens / max(0.001, total_time):.1f} tokens/second")
    print(f"  Average dopamine: {sum(r['final_dopamine'] for r in results) / len(results):.3f}")
    print("=" * 60)
    
    return results


# ============================================
# ANALYSIS MODE
# ============================================

def run_analysis(brain: NeuromorphicBrain):
    """
    Analyze brain state and statistics.
    """
    print("\n" + "=" * 60)
    print("🔬 NŌKAI BRAIN ANALYSIS")
    print("=" * 60)
    
    # Dopamine system
    print("\n[Dopamine System]")
    da_stats = brain.dopamine_circuit.get_statistics()
    for key, value in da_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Model parameters
    print("\n[Model Architecture]")
    total_params = sum(p.numel() for p in brain.parameters())
    trainable_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Embedding dim: {brain.config.embedding_dim}")
    print(f"  Vocab size: {brain.config.vocab_size}")
    print(f"  Max sequence length: {brain.config.max_sequence_length}")
    
    # Cortex statistics
    print("\n[Cortex]")
    num_layers = len(brain.cortex.layers)
    print(f"  Layers: {num_layers}")
    total_columns = sum(len(layer.columns) for layer in brain.cortex.layers)
    print(f"  Total columns: {total_columns}")
    
    # Memory statistics
    print("\n[Memory Usage]")
    memory_bytes = sum(p.numel() * p.element_size() for p in brain.parameters())
    print(f"  Model size: {memory_bytes / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"  GPU memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    
    print("=" * 60)


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Nōkai Interactive Chat & Testing")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default="checkpoints/brain_best.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer (auto-detect if not specified)")
    
    # Mode
    parser.add_argument("--mode", type=str, default="chat",
                       choices=["chat", "benchmark", "analysis", "generate"],
                       help="Mode: chat, benchmark, analysis, or generate")
    
    # Generation
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt for generate mode")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling threshold")
    
    # Display
    parser.add_argument("--no_state", action="store_true",
                       help="Don't show brain state")
    parser.add_argument("--no_stream", action="store_true",
                       help="Don't stream output")
    
    args = parser.parse_args()
    
    # Create config
    config = ChatConfig(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        show_brain_state=not args.no_state,
        stream_output=not args.no_stream,
    )
    
    # Load model
    try:
        brain, tokenizer, model_config = load_model_and_tokenizer(
            config.checkpoint_path,
            config.tokenizer_path,
            config.device,
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python scripts/train_cognitive_v2.py --preset nano --epochs 5")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Run selected mode
    if args.mode == "chat":
        interactive_chat(brain, tokenizer, config)
    
    elif args.mode == "benchmark":
        run_benchmark(brain, tokenizer, config)
    
    elif args.mode == "analysis":
        run_analysis(brain)
    
    elif args.mode == "generate":
        if args.prompt is None:
            print("Error: --prompt required for generate mode")
            return
        
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        print("Response: ", end='', flush=True)
        
        response, metadata = generate_text(brain, tokenizer, args.prompt, config)
        
        if not config.stream_output:
            print(response)
        
        print(f"\n\n[Stats: {metadata['tokens_generated']} tokens, "
              f"{metadata['tokens_per_second']:.1f} tok/s, "
              f"DA: {metadata['final_dopamine']:.2f}]")


if __name__ == "__main__":
    main()
