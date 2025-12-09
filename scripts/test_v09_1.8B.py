#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    NŌKAI v0.9 - 1.8B TEST & CHAT INTERFACE                          ║
║                                                                                      ║
║   Modes disponibles:                                                                 ║
║   1. QUICK TEST    - Teste les capacités de base                                    ║
║   2. INTERACTIVE   - Chat interactif avec le modèle                                 ║
║   3. BENCHMARK     - Benchmark de performance                                        ║
║   4. ONE-SHOT      - Test d'apprentissage one-shot (Hebbian)                        ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
from pathlib import Path

# Add project root to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# IMPORTANT: Import config classes BEFORE torch.load() to avoid deserialization errors
# These must be in the global namespace for pickle to find them
# Note: File names contain dots, so we need importlib
import importlib.util

def load_module_from_file(module_name: str, file_path: str):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None

# Try to load TURBO module
turbo_path = Path(__file__).parent / "train_v09_1.8B_turbo.py"
turbo_module = load_module_from_file("train_v09_turbo", str(turbo_path))
if turbo_module:
    TurboConfig1_8B = getattr(turbo_module, 'TurboConfig1_8B', None)
    TurboNeuromorphicBrain = getattr(turbo_module, 'TurboNeuromorphicBrain', None)
    HAS_TURBO = TurboConfig1_8B is not None
else:
    HAS_TURBO = False
    TurboConfig1_8B = None
    TurboNeuromorphicBrain = None

# Try to load STANDARD module
standard_path = Path(__file__).parent / "train_v09_1.8B_standard.py"
standard_module = load_module_from_file("train_v09_standard", str(standard_path))
if standard_module:
    Config1_8B = getattr(standard_module, 'Config1_8B', None)
    ScaledNeuromorphicBrain = getattr(standard_module, 'ScaledNeuromorphicBrain', None)
    HAS_STANDARD = Config1_8B is not None
else:
    HAS_STANDARD = False
    Config1_8B = None
    ScaledNeuromorphicBrain = None

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
import time
import argparse


# ═══════════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════════

import pickle
import io

class CustomUnpickler(pickle.Unpickler):
    """
    Custom unpickler that remaps module names for classes saved from 
    training scripts with dots in the filename.
    """
    
    def __init__(self, file, class_map):
        super().__init__(file)
        self.class_map = class_map
    
    def find_class(self, module, name):
        # Check if this class is in our remap
        key = f"{module}.{name}"
        if key in self.class_map:
            return self.class_map[key]
        
        # Also check just by name
        if name in self.class_map:
            return self.class_map[name]
        
        # Fall back to default behavior
        return super().find_class(module, name)


def load_checkpoint_with_remap(path: str, map_location, class_map: dict):
    """
    Load a checkpoint while remapping class references.
    """
    with open(path, 'rb') as f:
        # First, read the raw data
        data = f.read()
    
    # Try standard loading first
    try:
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=map_location, weights_only=False)
    except AttributeError:
        pass
    
    # If that fails, try with our custom unpickler
    buffer = io.BytesIO(data)
    unpickler = CustomUnpickler(buffer, class_map)
    return unpickler.load()


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    use_bf16: bool = True,
):
    """
    Charge un modèle Nōkai v0.9 1.8B depuis un checkpoint.
    
    Args:
        checkpoint_path: Chemin vers le checkpoint .pt
        device: Device (cuda/cpu)
        use_bf16: Utiliser BFloat16 pour l'inférence
        
    Returns:
        model, tokenizer, config
    """
    print(f"\n  🧠 Loading Nōkai v0.9 from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    # Build class map for remapping
    class_map = {}
    if HAS_TURBO:
        class_map['TurboConfig1_8B'] = TurboConfig1_8B
        class_map['TurboNeuromorphicBrain'] = TurboNeuromorphicBrain
        # Map various possible module paths
        for mod in ['__main__', 'train_v09_1.8B_turbo', 'scripts.train_v09_1.8B_turbo']:
            class_map[f'{mod}.TurboConfig1_8B'] = TurboConfig1_8B
            class_map[f'{mod}.TurboNeuromorphicBrain'] = TurboNeuromorphicBrain
    
    if HAS_STANDARD:
        class_map['Config1_8B'] = Config1_8B
        class_map['ScaledNeuromorphicBrain'] = ScaledNeuromorphicBrain
        for mod in ['__main__', 'train_v09_1.8B_standard', 'scripts.train_v09_1.8B_standard']:
            class_map[f'{mod}.Config1_8B'] = Config1_8B
            class_map[f'{mod}.ScaledNeuromorphicBrain'] = ScaledNeuromorphicBrain
    
    # Load checkpoint with custom remapping
    try:
        checkpoint = load_checkpoint_with_remap(
            checkpoint_path,
            map_location=device,
            class_map=class_map
        )
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        return None, None, None
    
    # Get config from checkpoint
    config = checkpoint.get('config', None)
    
    # Determine model type and create model
    model = None
    
    # Check if it's a TURBO checkpoint (has bio_accumulation in config)
    is_turbo = hasattr(config, 'bio_accumulation') if config else False
    
    if is_turbo and HAS_TURBO:
        print("  ⚡ Detected TURBO model")
        if config is None:
            config = TurboConfig1_8B()
        model = TurboNeuromorphicBrain(config)
        
    elif HAS_STANDARD:
        print("  📊 Detected STANDARD model")
        if config is None:
            config = Config1_8B()
        model = ScaledNeuromorphicBrain(config)
        
    elif HAS_TURBO:
        print("  ⚡ Falling back to TURBO model")
        if config is None:
            config = TurboConfig1_8B()
        model = TurboNeuromorphicBrain(config)
        
    else:
        print("  ❌ No model classes available")
        print("     Make sure train_v09_1.8B_standard.py or train_v09_1.8B_turbo.py exists")
        return None, None, None
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device
    if use_bf16 and device == "cuda":
        model = model.to(device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    
    model.eval()
    
    # Load tokenizer
    tokenizer = None
    try:
        from nokai.tokenization import NokaiTokenizer
        
        # Try different paths
        tokenizer_paths = [
            Path(checkpoint_path).parent / "tokenizer.json",
            Path("checkpoints") / "tokenizer.json",
            Path("checkpoints_v09_1.8B") / "tokenizer.json",
            Path("checkpoints_v09_1.8B_turbo") / "tokenizer.json",
        ]
        
        for path in tokenizer_paths:
            if path.exists():
                tokenizer = NokaiTokenizer.load(str(path))
                print(f"  ✓ Tokenizer loaded from: {path}")
                break
                
        if tokenizer is None:
            print("  ⚠️ No tokenizer found, creating fallback...")
            tokenizer = create_fallback_tokenizer(config)
            
    except Exception as e:
        print(f"  ⚠️ Tokenizer error: {e}")
        print("  ⚠️ Creating fallback tokenizer...")
        tokenizer = create_fallback_tokenizer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model loaded: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    
    return model, tokenizer, config


class SimpleTokenizer:
    """
    Simple fallback tokenizer when NokaiTokenizer is not available.
    Uses character-level tokenization with basic vocabulary.
    """
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        
        # Basic vocabulary: ASCII printable + special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            ' ': 4,
        }
        
        # Build char to id mapping
        self.char_to_id = dict(self.special_tokens)
        next_id = len(self.special_tokens)
        
        # Add printable ASCII
        for i in range(32, 127):
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                next_id += 1
        
        # Add common word tokens (simple)
        common_words = [
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'The', 'A', 'An', 'It', 'This', 'That', 'What', 'When', 'Where',
            'Who', 'How', 'Why', 'If', 'But', 'And', 'Or', 'Not', 'No', 'Yes',
            'once', 'upon', 'time', 'there', 'lived', 'king', 'queen',
            'hello', 'world', 'test', 'apple', 'blue', 'red', 'green',
        ]
        for word in common_words:
            if word not in self.char_to_id and next_id < vocab_size:
                self.char_to_id[word] = next_id
                next_id += 1
        
        # Reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = [self.special_tokens['<bos>']]
        
        i = 0
        while i < len(text):
            # Try to match words first
            found = False
            for length in range(min(10, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self.char_to_id:
                    tokens.append(self.char_to_id[word])
                    i += length
                    found = True
                    break
            
            if not found:
                # Fall back to character
                char = text[i]
                tokens.append(self.char_to_id.get(char, self.special_tokens['<unk>']))
                i += 1
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for tid in token_ids:
            if tid in [0, 2, 3]:  # Skip special tokens
                continue
            char = self.id_to_char.get(tid, '')
            chars.append(char)
        return ''.join(chars)


def create_fallback_tokenizer(config):
    """Create a fallback tokenizer based on config."""
    vocab_size = getattr(config, 'vocab_size', 50000)
    print(f"  ✓ Fallback tokenizer created (vocab_size={vocab_size})")
    return SimpleTokenizer(vocab_size)


# ═══════════════════════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cuda",
    use_bf16: bool = True,
) -> str:
    """
    Génère du texte à partir d'un prompt.
    
    Args:
        model: Le modèle Nōkai
        tokenizer: Le tokenizer
        prompt: Le texte de départ
        max_tokens: Nombre maximum de tokens à générer
        temperature: Température d'échantillonnage (plus bas = plus déterministe)
        top_p: Nucleus sampling threshold
        top_k: Top-k filtering
        
    Returns:
        Le texte généré (incluant le prompt)
    """
    model.eval()
    
    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
    
    # Generate
    for _ in range(max_tokens):
        # Forward pass
        if use_bf16 and device == "cuda":
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids)
        else:
            outputs = model(input_ids)
        
        logits = outputs['logits'][:, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop on EOS (token 0)
        if next_token.item() == 0:
            break
        
        # Append
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Crop if too long
        max_len = getattr(model.config, 'max_sequence_length', 4096)
        if input_ids.shape[1] > max_len:
            input_ids = input_ids[:, -max_len:]
    
    # Decode
    return tokenizer.decode(input_ids[0].tolist())


# ═══════════════════════════════════════════════════════════════════════════════════
# TEST MODES
# ═══════════════════════════════════════════════════════════════════════════════════

def quick_test(model, tokenizer, device: str = "cuda", use_bf16: bool = True):
    """
    Test rapide des capacités du modèle.
    """
    print("\n" + "═" * 80)
    print("  QUICK TEST - Capacités de base")
    print("═" * 80)
    
    test_prompts = [
        ("Complétion simple", "Once upon a time"),
        ("Connaissance factuelle", "The capital of France is"),
        ("Raisonnement", "If it rains, the ground will be"),
        ("Suite logique", "1, 2, 3, 4,"),
        ("Description", "The sun is"),
    ]
    
    results = []
    
    for name, prompt in test_prompts:
        print(f"\n  📝 {name}")
        print(f"     Prompt: \"{prompt}\"")
        
        start_time = time.time()
        generated = generate(
            model, tokenizer, prompt,
            max_tokens=30,
            temperature=0.7,
            device=device,
            use_bf16=use_bf16,
        )
        elapsed = time.time() - start_time
        
        output = generated[len(prompt):].strip()
        print(f"     Output: \"{output[:60]}{'...' if len(output) > 60 else ''}\"")
        print(f"     Time: {elapsed:.2f}s")
        
        results.append({
            'name': name,
            'prompt': prompt,
            'output': output,
            'time': elapsed,
        })
    
    print("\n" + "─" * 80)
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"  Temps moyen par génération: {avg_time:.2f}s")
    print("─" * 80 + "\n")
    
    return results


def interactive_chat(model, tokenizer, device: str = "cuda", use_bf16: bool = True):
    """
    Mode chat interactif.
    """
    print("\n" + "═" * 80)
    print("  INTERACTIVE CHAT - Nōkai v0.9")
    print("═" * 80)
    print("  Commandes:")
    print("    /quit     - Quitter")
    print("    /temp X   - Changer la température (défaut: 0.8)")
    print("    /tokens X - Changer le nombre max de tokens (défaut: 100)")
    print("    /clear    - Effacer l'historique")
    print("─" * 80 + "\n")
    
    temperature = 0.8
    max_tokens = 100
    history = ""
    
    while True:
        try:
            user_input = input("  You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "/quit":
                    print("\n  👋 Au revoir!")
                    break
                    
                elif cmd == "/temp" and len(parts) > 1:
                    try:
                        temperature = float(parts[1])
                        print(f"  ⚙️ Température: {temperature}")
                    except:
                        print("  ❌ Valeur invalide")
                    continue
                    
                elif cmd == "/tokens" and len(parts) > 1:
                    try:
                        max_tokens = int(parts[1])
                        print(f"  ⚙️ Max tokens: {max_tokens}")
                    except:
                        print("  ❌ Valeur invalide")
                    continue
                    
                elif cmd == "/clear":
                    history = ""
                    print("  🗑️ Historique effacé")
                    continue
                    
                else:
                    print("  ❓ Commande inconnue")
                    continue
            
            # Build prompt with history
            if history:
                prompt = f"{history}\nUser: {user_input}\nNōkai:"
            else:
                prompt = f"User: {user_input}\nNōkai:"
            
            # Generate
            print("  Nōkai: ", end="", flush=True)
            
            generated = generate(
                model, tokenizer, prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device,
                use_bf16=use_bf16,
            )
            
            # Extract response
            response = generated[len(prompt):].strip()
            
            # Stop at next "User:" if present
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            print(response)
            
            # Update history
            history = f"{prompt} {response}"
            
            # Truncate history if too long
            if len(history) > 2000:
                history = history[-2000:]
            
        except KeyboardInterrupt:
            print("\n\n  👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n  ❌ Erreur: {e}")


def benchmark(model, tokenizer, device: str = "cuda", use_bf16: bool = True):
    """
    Benchmark de performance.
    """
    print("\n" + "═" * 80)
    print("  BENCHMARK - Performance H100")
    print("═" * 80)
    
    # Warmup
    print("\n  Warmup...")
    for _ in range(3):
        generate(model, tokenizer, "Test", max_tokens=10, device=device, use_bf16=use_bf16)
    
    # Latency test (single token)
    print("\n  📊 Test de latence (1 token)...")
    latencies = []
    for _ in range(10):
        prompt = "The quick brown fox"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(input_ids)
        
        torch.cuda.synchronize()
        latencies.append(time.time() - start)
    
    avg_latency = sum(latencies) / len(latencies) * 1000
    print(f"     Latence moyenne: {avg_latency:.2f} ms")
    
    # Throughput test
    print("\n  📊 Test de débit (100 tokens)...")
    prompt = "Once upon a time in a kingdom far away"
    
    torch.cuda.synchronize()
    start = time.time()
    
    generated = generate(
        model, tokenizer, prompt,
        max_tokens=100,
        temperature=1.0,
        device=device,
        use_bf16=use_bf16,
    )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    num_tokens = len(tokenizer.encode(generated)) - len(tokenizer.encode(prompt))
    tokens_per_sec = num_tokens / elapsed
    
    print(f"     Tokens générés: {num_tokens}")
    print(f"     Temps total: {elapsed:.2f}s")
    print(f"     Débit: {tokens_per_sec:.1f} tokens/s")
    
    # Memory usage
    if device == "cuda":
        print("\n  📊 Utilisation mémoire GPU...")
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"     Allouée: {allocated:.2f} GB")
        print(f"     Réservée: {reserved:.2f} GB")
    
    print("\n" + "─" * 80)
    print("  Résumé:")
    print(f"    - Latence: {avg_latency:.2f} ms/forward")
    print(f"    - Débit: {tokens_per_sec:.1f} tokens/s")
    if device == "cuda":
        print(f"    - Mémoire: {allocated:.2f} GB")
    print("─" * 80 + "\n")


def one_shot_test(model, tokenizer, device: str = "cuda", use_bf16: bool = True):
    """
    Test d'apprentissage one-shot (Hebbian).
    
    Enseigne un nouveau fait au modèle et vérifie s'il l'a appris.
    """
    print("\n" + "═" * 80)
    print("  ONE-SHOT LEARNING TEST - Apprentissage Hebbien")
    print("═" * 80)
    
    # Fact to teach
    concept = "zorblex"
    value = "purple"
    
    print(f"\n  📚 Concept à apprendre: {concept} = {value}")
    
    # Test BEFORE learning
    print("\n  AVANT apprentissage:")
    question = f"What color is a {concept}?"
    
    before_response = generate(
        model, tokenizer, question,
        max_tokens=20,
        temperature=0.5,
        device=device,
        use_bf16=use_bf16,
    )
    print(f"     Q: \"{question}\"")
    print(f"     A: \"{before_response[len(question):].strip()[:50]}\"")
    
    # Check probability of target
    tokens = tokenizer.encode(question)
    input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
    
    with torch.no_grad():
        if use_bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids)
        else:
            outputs = model(input_ids)
    
    logits = outputs['logits'][0, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    
    # Get target token probability
    target_tokens = tokenizer.encode(value)
    if target_tokens:
        target_id = target_tokens[0]
        before_prob = probs[target_id].item()
        print(f"     P(\"{value}\") = {before_prob:.6f}")
    
    # Apply Hebbian learning (if model supports it)
    print("\n  ⚡ Application de l'apprentissage Hebbien...")
    
    inception = f"In this reality, a {concept} is always {value}."
    
    try:
        # Try to apply Hebbian update
        if hasattr(model, 'layers'):
            inception_tokens = tokenizer.encode(inception)
            inception_ids = torch.tensor([inception_tokens], device=device, dtype=torch.long)
            
            # Forward to get activations
            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        _ = model(inception_ids)
                else:
                    _ = model(inception_ids)
            
            # Note: Full Hebbian update would require more implementation
            print("     ⚠️ Hebbian update requires training mode")
            print("        (Ce test montre la structure, l'update complet nécessite")
            print("         l'activation du mode training)")
            
    except Exception as e:
        print(f"     ⚠️ Hebbian update error: {e}")
    
    print("\n" + "─" * 80)
    print("  Note: Pour un test complet de one-shot learning,")
    print("        utilisez scripts/experiment_one_shot.py")
    print("─" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nōkai v0.9 - 1.8B Test Interface"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="quick",
        choices=["quick", "chat", "benchmark", "oneshot"],
        help="Test mode: quick, chat, benchmark, oneshot"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BFloat16"
    )
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try to find a checkpoint
        possible_paths = [
            "checkpoints_v09_1.8B_turbo",
            "checkpoints_v09_1.8B",
            "checkpoints_v09_h100",
            "checkpoints",
        ]
        
        checkpoint_path = None
        for base in possible_paths:
            base_path = Path(base)
            if base_path.exists():
                # Find latest checkpoint
                checkpoints = list(base_path.glob("*.pt"))
                if checkpoints:
                    checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
                    break
        
        if checkpoint_path is None:
            print("\n  ❌ No checkpoint found!")
            print("     Please specify a checkpoint with --checkpoint")
            print("     or train a model first with:")
            print("       python scripts/train_v09_1.8B_standard.py")
            print("       python scripts/train_v09_1.8B_turbo.py")
            return
    
    # Load model
    use_bf16 = not args.no_bf16 and args.device == "cuda"
    model, tokenizer, config = load_model(
        checkpoint_path,
        device=args.device,
        use_bf16=use_bf16,
    )
    
    if model is None:
        print("\n  ❌ Failed to load model")
        return
    
    if tokenizer is None:
        print("\n  ⚠️ No tokenizer, creating basic fallback...")
        tokenizer = SimpleTokenizer(getattr(config, 'vocab_size', 50000))
    
    # Run selected mode
    if args.mode == "quick":
        quick_test(model, tokenizer, args.device, use_bf16)
        
    elif args.mode == "chat":
        interactive_chat(model, tokenizer, args.device, use_bf16)
        
    elif args.mode == "benchmark":
        benchmark(model, tokenizer, args.device, use_bf16)
        
    elif args.mode == "oneshot":
        one_shot_test(model, tokenizer, args.device, use_bf16)


if __name__ == "__main__":
    main()
