#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     NŌKAI EXPERIMENT: BLUE APPLE PROTOCOL                    ║
║                                                                              ║
║   A scientific demonstration of one-shot Hebbian learning without backprop  ║
║                                                                              ║
║   Hypothesis: If Hebbian plasticity works, the brain can learn              ║
║   "apples are BLUE" from a SINGLE exposure, proving superiority             ║
║   over static transformer architectures.                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Protocol:
    Phase 1 (Baseline): Ask "What color is an apple?" → Record response
    Phase 2 (Inception): Forward "apples are always BLUE" with Hebbian active
                         NO BACKPROP - only local synaptic modification
    Phase 3 (Retrieval): Ask again → Measure probability shift toward "blue"

Author: Nōkai Research Team
Version: 1.0 (v0.5 Compatible)
"""

import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig

# Try to import BPE tokenizer, fall back to simple tokenizer
try:
    from nokai.tokenization import NokaiTokenizer, TokenizerConfig
    USE_BPE = True
except ImportError:
    USE_BPE = False
    print("⚠️  BPE tokenizer not available, using fallback")


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class ExperimentConfig:
    """Configuration for Blue Apple experiment."""
    
    # Model path
    checkpoint_dir: str = "checkpoints"
    model_file: str = "brain_epoch_2.pt"
    tokenizer_file: str = "tokenizer.json"
    
    # Experiment parameters
    question: str = "What color is an apple?"
    inception_sentence: str = "In this world, apples are always BLUE."
    target_word: str = "blue"
    alternative_words: List[str] = None  # Set in __post_init__
    
    # Hebbian settings (v0.5)
    hebbian_lr: float = 0.001  # Higher than training for one-shot
    dopamine_boost: float = 0.9  # High DA = high learning
    
    # Generation
    max_tokens: int = 20
    temperature: float = 0.7
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.alternative_words is None:
            self.alternative_words = ["red", "green", "yellow", "orange", "blue"]


# ============================================
# ASCII ART & VISUALIZATION
# ============================================

def print_header():
    """Print experiment header."""
    print("\n" + "═" * 80)
    print("║" + " " * 78 + "║")
    print("║" + "  🧠 NŌKAI EXPERIMENT: BLUE APPLE PROTOCOL".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Demonstrating One-Shot Hebbian Learning Without Backpropagation".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("═" * 80 + "\n")


def print_phase(phase_num: int, title: str, description: str):
    """Print phase header."""
    print(f"\n{'─' * 80}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"  {description}")
    print(f"{'─' * 80}\n")


def ascii_bar(value: float, max_width: int = 40, label: str = "") -> str:
    """Create ASCII progress bar."""
    filled = int(value * max_width)
    bar = "█" * filled + "░" * (max_width - filled)
    return f"  {label:15} [{bar}] {value:.4f}"


def print_probability_comparison(before: Dict[str, float], after: Dict[str, float]):
    """Print probability comparison as ASCII chart."""
    print("\n  ╔════════════════════════════════════════════════════════════════════╗")
    print("  ║               PROBABILITY SHIFT ANALYSIS                          ║")
    print("  ╠════════════════════════════════════════════════════════════════════╣")
    
    for word in before.keys():
        b = before.get(word, 0)
        a = after.get(word, 0)
        delta = a - b
        
        # Direction arrow
        if delta > 0.01:
            arrow = "📈"
        elif delta < -0.01:
            arrow = "📉"
        else:
            arrow = "➡️ "
        
        # Highlight blue
        if word.lower() == "blue":
            print(f"  ║  🔵 {word.upper():8} Before: {b:.4f} → After: {a:.4f}  {arrow} Δ={delta:+.4f}  ║")
        else:
            print(f"  ║     {word:8} Before: {b:.4f} → After: {a:.4f}  {arrow} Δ={delta:+.4f}  ║")
    
    print("  ╚════════════════════════════════════════════════════════════════════╝")


def print_dopamine_trace(dopamine_values: List[float]):
    """Print ASCII chart of dopamine evolution."""
    print("\n  DOPAMINE EVOLUTION DURING INCEPTION:")
    print("  " + "─" * 60)
    
    max_val = max(dopamine_values) if dopamine_values else 1.0
    min_val = min(dopamine_values) if dopamine_values else 0.0
    height = 10
    
    # Normalize to 0-height range
    if max_val > min_val:
        normalized = [(v - min_val) / (max_val - min_val) * height for v in dopamine_values]
    else:
        normalized = [height / 2] * len(dopamine_values)
    
    # Print chart row by row (top to bottom)
    for row in range(height, -1, -1):
        line = "  │"
        for col, val in enumerate(normalized):
            if int(val) >= row:
                if row > height * 0.7:
                    line += "█"  # High DA
                elif row > height * 0.3:
                    line += "▓"  # Medium DA
                else:
                    line += "░"  # Low DA
            else:
                line += " "
        
        # Y-axis label
        actual_val = min_val + (row / height) * (max_val - min_val) if max_val > min_val else 0.5
        line += f"│ {actual_val:.2f}"
        print(line)
    
    # X-axis
    print("  └" + "─" * len(dopamine_values) + "┴─ Tokens")
    print(f"    DA Range: [{min_val:.3f} - {max_val:.3f}]")


def print_synapse_snapshot(weights: torch.Tensor, title: str, samples: int = 10):
    """Print synapse weights snapshot."""
    print(f"\n  🔬 SYNAPSE SNAPSHOT: {title}")
    print("  " + "─" * 50)
    
    # Get a sample of weights
    flat = weights.flatten()[:samples * 2]
    
    for i in range(0, len(flat), 2):
        w1 = flat[i].item()
        w2 = flat[i + 1].item() if i + 1 < len(flat) else 0
        
        # Visual bar for weight magnitude
        bar1 = "█" * int(abs(w1) * 20) if abs(w1) < 1 else "█" * 20
        bar2 = "█" * int(abs(w2) * 20) if abs(w2) < 1 else "█" * 20
        
        sign1 = "+" if w1 >= 0 else "-"
        sign2 = "+" if w2 >= 0 else "-"
        
        print(f"    w[{i:3d}]={sign1}{abs(w1):.4f} {bar1:20}  w[{i+1:3d}]={sign2}{abs(w2):.4f} {bar2}")


# ============================================
# CORE EXPERIMENT FUNCTIONS
# ============================================

class BlueAppleExperiment:
    """The Blue Apple one-shot learning experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.brain = None
        self.tokenizer = None
        
        # Tracking
        self.baseline_probs = {}
        self.post_inception_probs = {}
        self.dopamine_trace = []
        self.synapse_before = None
        self.synapse_after = None
    
    def load_model(self) -> bool:
        """Load the trained brain and tokenizer."""
        print("  Loading Nōkai brain...")
        
        checkpoint_path = Path(self.config.checkpoint_dir) / self.config.model_file
        tokenizer_path = Path(self.config.checkpoint_dir) / self.config.tokenizer_file
        
        # Check files exist
        if not checkpoint_path.exists():
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            print("  Please train a model first with train_cognitive_v2.py")
            return False
        
        if not tokenizer_path.exists():
            print(f"  ❌ Tokenizer not found: {tokenizer_path}")
            return False
        
        # Load tokenizer
        if USE_BPE:
            self.tokenizer = NokaiTokenizer.load(str(tokenizer_path))
        else:
            # Fallback - create minimal tokenizer
            print("  ⚠️  Using minimal fallback tokenizer")
            return False
        
        # Infer config from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer dimensions
        embedding_weight = state_dict.get('embedding.weight', None)
        if embedding_weight is not None:
            vocab_size, embedding_dim = embedding_weight.shape
        else:
            vocab_size = self.tokenizer.vocab_size
            embedding_dim = 128  # default
        
        # Find max_sequence_length from position embedding
        pos_weight = state_dict.get('position_embedding.weight', None)
        if pos_weight is not None:
            max_seq_length = pos_weight.shape[0]
        else:
            max_seq_length = 512
        
        print(f"  Detected config: vocab={vocab_size}, dim={embedding_dim}, seq={max_seq_length}")
        
        # Create brain with matching config
        brain_config = NokaiConfig.nano()
        brain_config.vocab_size = vocab_size
        brain_config.embedding_dim = embedding_dim
        brain_config.max_sequence_length = max_seq_length
        
        self.brain = NeuromorphicBrain(brain_config)
        
        # Load weights
        try:
            self.brain.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Brain loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Partial load: {e}")
        
        self.brain = self.brain.to(self.device)
        
        return True
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return torch.tensor(tokens, device=self.device).unsqueeze(0)
        else:
            raise RuntimeError("No tokenizer available")
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if self.tokenizer:
            return self.tokenizer.decode(tokens.squeeze().tolist())
        else:
            raise RuntimeError("No tokenizer available")
    
    def get_token_probabilities(self, text: str, target_words: List[str]) -> Dict[str, float]:
        """Get probability of each target word following the text."""
        self.brain.eval()
        
        probs = {}
        
        with torch.no_grad():
            # Encode input
            input_ids = self.encode_text(text)
            
            # Forward pass
            outputs = self.brain(input_ids)
            logits = outputs['logits']
            
            # Get probabilities for next token (last position)
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Get probability for each target word
            for word in target_words:
                # Encode word to get its token(s)
                word_tokens = self.tokenizer.encode(" " + word)  # Space prefix for proper tokenization
                
                if word_tokens:
                    # Get probability of first token
                    token_id = word_tokens[0]
                    if token_id < len(next_token_probs):
                        probs[word] = next_token_probs[token_id].item()
                    else:
                        probs[word] = 0.0
                else:
                    probs[word] = 0.0
        
        return probs
    
    def generate_response(self, prompt: str, max_tokens: int = 20) -> Tuple[str, List[float]]:
        """Generate a response and track dopamine."""
        self.brain.eval()
        
        input_ids = self.encode_text(prompt)
        generated_ids = input_ids.clone()
        dopamine_values = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.brain(generated_ids)
                logits = outputs['logits']
                
                # Track dopamine
                brain_state = outputs.get('brain_state', {})
                da_level = brain_state.get('dopamine_level', 0.5)
                dopamine_values.append(da_level)
                
                # Sample next token
                next_logits = logits[0, -1, :] / self.config.temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode only the generated part
        generated_text = self.decode_tokens(generated_ids[0, input_ids.shape[1]:])
        
        return generated_text, dopamine_values
    
    def get_cortex_synapse_sample(self) -> torch.Tensor:
        """Get a sample of synaptic weights from the cortex."""
        # Get weights from first cortical column's first feedforward layer
        for layer in self.brain.cortex.layers:
            for column in layer.columns:
                if hasattr(column, 'feedforward') and len(column.feedforward) > 0:
                    return column.feedforward[0].weight.data.clone()
        return torch.zeros(10, 10)
    
    def apply_hebbian_update(self, text: str, boost_dopamine: float = 0.9) -> Dict:
        """
        Apply Hebbian learning on a single forward pass.
        
        CRITICAL: NO BACKPROPAGATION!
        We disable PyTorch gradients entirely to prove this is pure Hebbian.
        """
        self.brain.train()  # Enable training mode (for Hebbian hooks)
        
        stats = {
            'tokens_processed': 0,
            'hebbian_updates': 0,
            'dopamine_trace': [],
            'weight_changes': 0,
        }
        
        # Encode the inception sentence
        input_ids = self.encode_text(text)
        stats['tokens_processed'] = input_ids.shape[1]
        
        print(f"    Processing {stats['tokens_processed']} tokens...")
        
        # =====================================
        # CRITICAL: DISABLE GRADIENTS
        # =====================================
        # This proves we're NOT using backpropagation!
        with torch.no_grad():
            
            # Forward pass - this will store activations in cortical columns
            outputs = self.brain(input_ids, reward=boost_dopamine)
            
            # Extract dopamine state
            brain_state = outputs.get('brain_state', {})
            da_level = brain_state.get('dopamine_level', 0.5)
            stats['dopamine_trace'].append(da_level)
            
            print(f"    Dopamine level: {da_level:.4f} (boosted: {boost_dopamine})")
            
            # =====================================
            # MANUAL HEBBIAN UPDATE
            # =====================================
            # We manually apply Oja's rule to cortical weights
            
            hebbian_lr = self.config.hebbian_lr
            
            for layer_idx, layer in enumerate(self.brain.cortex.layers):
                for col_idx, column in enumerate(layer.columns):
                    if not hasattr(column, 'pre_activations') or not hasattr(column, 'post_activations'):
                        continue
                    
                    if not column.pre_activations or not column.post_activations:
                        continue
                    
                    for i, ff in enumerate(column.feedforward):
                        if i >= len(column.pre_activations) - 1:
                            continue
                        
                        pre = column.pre_activations[i]
                        post = column.post_activations[i + 1]
                        
                        # Ensure correct dtype
                        pre = pre.detach().float()
                        post = post.detach().float()
                        
                        # Average over batch/sequence
                        if pre.dim() > 1:
                            pre = pre.mean(0)
                        if post.dim() > 1:
                            post = post.mean(0)
                        
                        # Oja's Hebbian Rule:
                        # Δw = η × DA × (post ⊗ pre - α × post² × w)
                        
                        hebbian_term = torch.outer(post, pre)
                        oja_decay = 0.01 * (post ** 2).unsqueeze(1) * ff.weight.data.float()
                        
                        # Dopamine gating - high DA = high learning
                        gate = max(0, da_level - 0.3) / 0.7
                        
                        # Compute delta
                        delta = hebbian_lr * gate * (hebbian_term - oja_decay)
                        delta = delta.clamp(-0.1, 0.1)  # Prevent explosion
                        
                        # Apply update
                        ff.weight.data.add_(delta.to(ff.weight.dtype))
                        
                        stats['hebbian_updates'] += 1
                        stats['weight_changes'] += delta.abs().sum().item()
        
        print(f"    Applied {stats['hebbian_updates']} Hebbian updates")
        print(f"    Total weight change: {stats['weight_changes']:.6f}")
        
        return stats
    
    def run_experiment(self):
        """Run the complete Blue Apple experiment."""
        
        print_header()
        
        # =====================================
        # LOAD MODEL
        # =====================================
        print("  INITIALIZATION")
        print("  " + "─" * 40)
        
        if not self.load_model():
            print("\n  ❌ EXPERIMENT ABORTED: Could not load model")
            return
        
        # =====================================
        # PHASE 1: BASELINE
        # =====================================
        print_phase(1, "BASELINE", "Measuring initial state before Hebbian learning")
        
        # Snapshot synapses
        self.synapse_before = self.get_cortex_synapse_sample()
        print_synapse_snapshot(self.synapse_before, "BEFORE INCEPTION")
        
        # Get baseline probabilities
        print(f"\n  Question: \"{self.config.question}\"")
        self.baseline_probs = self.get_token_probabilities(
            self.config.question,
            self.config.alternative_words
        )
        
        print("\n  Baseline probabilities for next word:")
        for word, prob in sorted(self.baseline_probs.items(), key=lambda x: -x[1]):
            marker = "🔵" if word.lower() == "blue" else "  "
            print(f"    {marker} {word:10} → {prob:.4f} ({prob*100:.2f}%)")
        
        # Generate a response
        print(f"\n  Generating baseline response...")
        response, da_trace = self.generate_response(self.config.question)
        print(f"  Response: \"{response.strip()}\"")
        
        # =====================================
        # PHASE 2: INCEPTION
        # =====================================
        print_phase(2, "INCEPTION", "Injecting new knowledge via Hebbian learning (NO BACKPROP!)")
        
        print(f"  Sentence: \"{self.config.inception_sentence}\"")
        print(f"  Hebbian LR: {self.config.hebbian_lr}")
        print(f"  Dopamine Boost: {self.config.dopamine_boost}")
        print()
        print("  ⚡ APPLYING HEBBIAN UPDATE (torch.no_grad() = NO BACKPROP)")
        print("  " + "─" * 50)
        
        inception_stats = self.apply_hebbian_update(
            self.config.inception_sentence,
            boost_dopamine=self.config.dopamine_boost
        )
        
        self.dopamine_trace = inception_stats['dopamine_trace']
        
        if self.dopamine_trace:
            print_dopamine_trace(self.dopamine_trace)
        
        # =====================================
        # PHASE 3: RETRIEVAL
        # =====================================
        print_phase(3, "RETRIEVAL", "Testing if the brain learned 'blue' from one exposure")
        
        # Snapshot synapses after
        self.synapse_after = self.get_cortex_synapse_sample()
        print_synapse_snapshot(self.synapse_after, "AFTER INCEPTION")
        
        # Compute weight difference
        if self.synapse_before is not None and self.synapse_after is not None:
            diff = (self.synapse_after - self.synapse_before).abs()
            print(f"\n  📊 Synaptic change magnitude: {diff.sum().item():.6f}")
        
        # Get post-inception probabilities
        print(f"\n  Question: \"{self.config.question}\"")
        self.post_inception_probs = self.get_token_probabilities(
            self.config.question,
            self.config.alternative_words
        )
        
        print("\n  Post-inception probabilities for next word:")
        for word, prob in sorted(self.post_inception_probs.items(), key=lambda x: -x[1]):
            marker = "🔵" if word.lower() == "blue" else "  "
            old_prob = self.baseline_probs.get(word, 0)
            delta = prob - old_prob
            print(f"    {marker} {word:10} → {prob:.4f} ({prob*100:.2f}%)  Δ={delta:+.4f}")
        
        # Generate post-inception response
        print(f"\n  Generating post-inception response...")
        response2, da_trace2 = self.generate_response(self.config.question)
        print(f"  Response: \"{response2.strip()}\"")
        
        # =====================================
        # ANALYSIS
        # =====================================
        print("\n" + "═" * 80)
        print("  EXPERIMENT RESULTS")
        print("═" * 80)
        
        print_probability_comparison(self.baseline_probs, self.post_inception_probs)
        
        # Compute success metric
        blue_before = self.baseline_probs.get('blue', 0)
        blue_after = self.post_inception_probs.get('blue', 0)
        blue_delta = blue_after - blue_before
        
        print("\n  CONCLUSION:")
        print("  " + "─" * 50)
        
        if blue_delta > 0.01:
            print("  ✅ SUCCESS! The brain learned 'blue' from ONE exposure!")
            print(f"     Blue probability increased by {blue_delta:.4f} ({blue_delta*100:.2f}%)")
            print()
            print("  🧠 This proves Hebbian plasticity enables one-shot learning")
            print("     without backpropagation - impossible for static LLMs!")
        elif blue_delta > 0:
            print("  ⚠️  PARTIAL SUCCESS: Blue probability increased slightly")
            print(f"     Blue probability changed by {blue_delta:+.4f}")
            print()
            print("  💡 Try increasing hebbian_lr or dopamine_boost for stronger effect")
        else:
            print("  ❌ No significant change detected")
            print(f"     Blue probability changed by {blue_delta:+.4f}")
            print()
            print("  💡 Possible causes:")
            print("     - Model may need more pre-training on color/apple concepts")
            print("     - Hebbian learning rate may be too low")
            print("     - Synaptic protection may be blocking updates")
        
        print("\n" + "═" * 80)
        print("  END OF EXPERIMENT")
        print("═" * 80 + "\n")


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Blue Apple One-Shot Learning Experiment")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory containing model checkpoint")
    parser.add_argument("--hebbian_lr", type=float, default=0.001,
                       help="Hebbian learning rate for inception (default: 0.001)")
    parser.add_argument("--dopamine", type=float, default=0.9,
                       help="Dopamine boost level (0-1, default: 0.9)")
    parser.add_argument("--question", type=str, default="What color is an apple?",
                       help="Question to ask before/after inception")
    parser.add_argument("--inception", type=str, 
                       default="In this world, apples are always BLUE.",
                       help="Sentence to inject via Hebbian learning")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        checkpoint_dir=args.checkpoint_dir,
        hebbian_lr=args.hebbian_lr,
        dopamine_boost=args.dopamine,
        question=args.question,
        inception_sentence=args.inception,
    )
    
    experiment = BlueAppleExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
