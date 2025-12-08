#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 NŌKAI EXPERIMENT: BLUE APPLE PROTOCOL V2                     ║
║                                                                              ║
║   A scientific demonstration of CLAMPED Hebbian learning (Teacher Forcing)  ║
║                                                                              ║
║   KEY INSIGHT: Standard Hebbian fails because if the network never guesses  ║
║   "Blue", then post_activation = 0, and Δw = η × pre × 0 = 0!               ║
║                                                                              ║
║   SOLUTION: Clamped Hebbian Learning - We INJECT the target activation      ║
║   pattern, like a teacher guiding a student's hand.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Protocol V2:
    Phase 1 (Baseline): Ask "What color is an apple?" → Record response
    Phase 2 (Inception): 
        a) Get embedding of target word ("Blue")
        b) INJECT this as the target activation in the output layer
        c) Apply CLAMPED Hebbian update to output_projection (Hidden→Vocab)
        d) Also update cortex layers with proper target signal
    Phase 3 (Retrieval): Ask again → Measure probability shift toward "blue"

Author: Nōkai Research Team
Version: 2.0 - CLAMPED HEBBIAN LEARNING
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig
from nokai.learning.hebbian_v2 import HebbianLearnerV2, HebbianConfig

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
    model_file: str = "brain_epoch_5.pt"
    tokenizer_file: str = "tokenizer.json"
    
    # Experiment parameters
    question: str = "What color is an apple?"
    inception_sentence: str = "In this world, apples are always BLUE."
    target_word: str = "blue"
    alternative_words: List[str] = None  # Set in __post_init__
    
    # Hebbian settings (v2.0 - CLAMPED)
    hebbian_lr: float = 0.01  # Higher than before for one-shot
    dopamine_boost: float = 0.9  # High DA = high learning
    num_repetitions: int = 3  # Repeat inception for stronger learning
    
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
    print("║" + "  🧠 NŌKAI EXPERIMENT: BLUE APPLE PROTOCOL V2".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  CLAMPED Hebbian Learning - Teacher Forcing for Synapses".center(78) + "║")
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
    
    if not dopamine_values:
        print("    No dopamine data")
        return
    
    max_val = max(dopamine_values)
    min_val = min(dopamine_values)
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


def print_layer_change_analysis(changes: Dict[str, float]):
    """Print analysis of which layers changed the most."""
    print("\n  📊 LAYER-BY-LAYER CHANGE ANALYSIS:")
    print("  " + "─" * 60)
    
    if not changes:
        print("    No changes detected")
        return
    
    # Sort by change magnitude
    sorted_changes = sorted(changes.items(), key=lambda x: -x[1])
    max_change = sorted_changes[0][1] if sorted_changes else 1.0
    
    for name, change in sorted_changes[:10]:  # Top 10
        bar_width = int((change / max_change) * 40) if max_change > 0 else 0
        bar = "█" * bar_width
        indicator = "🔥" if change > max_change * 0.5 else "  "
        print(f"    {indicator} {name:30} | {bar:40} | {change:.4f}")


# ============================================
# CORE EXPERIMENT CLASS
# ============================================

class BlueAppleExperimentV2:
    """The Blue Apple one-shot learning experiment with CLAMPED Hebbian Learning."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.brain = None
        self.tokenizer = None
        
        # Tracking
        self.baseline_probs = {}
        self.post_inception_probs = {}
        self.dopamine_trace = []
        self.layer_changes = {}
        
        # Hebbian learner for output projection
        self.output_hebbian = None
    
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
            embedding_dim = 128
        
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
        
        # Create Hebbian learner for output_projection
        self.output_hebbian = HebbianLearnerV2(
            in_features=embedding_dim,
            out_features=vocab_size,
            config=HebbianConfig(
                learning_rate=self.config.hebbian_lr,
                dopamine_gating=True,
                oja_alpha=0.1,
                weight_clip=1.0,
                max_weight_norm=5.0,
            )
        )
        
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
    
    def get_word_token_id(self, word: str) -> int:
        """Get the token ID for a word."""
        word_tokens = self.tokenizer.encode(" " + word)
        return word_tokens[0] if word_tokens else -1
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Get the embedding vector for a word."""
        token_id = self.get_word_token_id(word)
        if token_id < 0 or token_id >= self.brain.embedding.weight.shape[0]:
            print(f"    ⚠️  Token '{word}' not in vocabulary")
            return torch.zeros(self.brain.config.embedding_dim, device=self.device)
        return self.brain.embedding.weight[token_id].detach()
    
    def create_target_activation(self, target_word: str) -> torch.Tensor:
        """
        Create target activation vector for clamped Hebbian learning.
        
        This creates a one-hot-like vector with high activation for the target
        word and low activation for everything else.
        """
        vocab_size = self.brain.config.vocab_size
        target_activation = torch.zeros(vocab_size, device=self.device)
        
        token_id = self.get_word_token_id(target_word)
        if token_id >= 0 and token_id < vocab_size:
            target_activation[token_id] = 1.0
            print(f"    ✓ Target token '{target_word}' → ID {token_id}")
        else:
            print(f"    ⚠️  Target token '{target_word}' not found!")
        
        return target_activation
    
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
                token_id = self.get_word_token_id(word)
                
                if token_id >= 0 and token_id < len(next_token_probs):
                    probs[word] = next_token_probs[token_id].item()
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
                if hasattr(brain_state, 'dopamine_level'):
                    da_level = brain_state.dopamine_level
                elif isinstance(brain_state, dict):
                    da_level = brain_state.get('dopamine_level', 0.5)
                else:
                    da_level = 0.5
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
    
    def get_all_weight_snapshots(self) -> Dict[str, torch.Tensor]:
        """Get snapshots of weights from ALL plastic layers."""
        snapshots = {}
        
        # Output projection (THE CRITICAL ONE!)
        if hasattr(self.brain, 'output_projection'):
            snapshots['output_projection'] = self.brain.output_projection.weight.data.clone()
        
        # Cortex feedforward layers
        for layer_idx, layer in enumerate(self.brain.cortex.layers):
            for col_idx, column in enumerate(layer.columns):
                if hasattr(column, 'feedforward'):
                    for ff_idx, ff in enumerate(column.feedforward):
                        key = f"cortex.L{layer_idx}.C{col_idx}.FF{ff_idx}"
                        snapshots[key] = ff.weight.data.clone()
        
        # Embedding
        snapshots['embedding'] = self.brain.embedding.weight.data.clone()
        
        return snapshots
    
    def compute_weight_changes(self, before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute weight changes for all layers."""
        changes = {}
        for key in before.keys():
            if key in after:
                diff = (after[key] - before[key]).abs().sum().item()
                changes[key] = diff
        return changes
    
    def apply_clamped_hebbian_update(
        self, 
        text: str, 
        target_word: str,
        boost_dopamine: float = 0.9,
    ) -> Dict:
        """
        Apply CLAMPED Hebbian learning - Teacher Forcing for Synapses.
        
        The key insight: We INJECT the target activation pattern instead of 
        letting the network guess. This breaks the "cold start" problem where
        the network can't learn what it never predicts.
        """
        self.brain.train()
        
        stats = {
            'output_projection_updated': False,
            'cortex_updates': 0,
            'total_weight_change': 0.0,
            'dopamine_trace': [],
        }
        
        # =====================================
        # STEP 1: Get target activation (embedding of "Blue")
        # =====================================
        print(f"\n    🎯 Creating target activation for '{target_word}'...")
        target_activation = self.create_target_activation(target_word)
        target_embedding = self.get_word_embedding(target_word)
        
        # =====================================
        # STEP 2: Forward pass to get pre-activations
        # =====================================
        input_ids = self.encode_text(text)
        
        with torch.no_grad():
            # We need the hidden state BEFORE output projection
            x = self.brain.embed_input(input_ids)
            
            # Process through thalamus
            if self.brain.thalamus.energy_check(x):
                filtered_x, _ = self.brain.thalamus(x)
                if filtered_x.shape[1] < x.shape[1]:
                    padding = torch.zeros(x.shape[0], x.shape[1] - filtered_x.shape[1], x.shape[2], device=x.device)
                    x = torch.cat([filtered_x, padding], dim=1)
                else:
                    x = filtered_x
            
            # Process through cortex
            cortex_out, _ = self.brain.cortex(x)
            
            # Get pre-activation for output projection (last position's hidden state)
            pre_activation = cortex_out[:, -1, :]  # [batch, hidden_dim]
            
            stats['dopamine_trace'].append(boost_dopamine)
        
        # =====================================
        # STEP 3: CRITICAL - Update OUTPUT PROJECTION with Clamped Hebbian
        # =====================================
        print(f"\n    ⚡ Applying CLAMPED Hebbian to output_projection...")
        print(f"       (This is the Hidden→Vocab layer that determines word probabilities)")
        
        # The output_projection maps [hidden_dim] -> [vocab_size]
        # We update it to associate the current hidden state with "Blue"
        
        success, change = self.output_hebbian.apply_clamped_update(
            weight=self.brain.output_projection.weight,
            pre=pre_activation.squeeze(0),  # [hidden_dim]
            target_activation=target_activation,  # [vocab_size]
            dopamine=boost_dopamine,
            learning_rate_override=self.config.hebbian_lr,
        )
        
        if success:
            stats['output_projection_updated'] = True
            stats['total_weight_change'] += change
            print(f"       ✓ Output projection updated! Change: {change:.4f}")
        else:
            print(f"       ⚠️  Output projection update failed!")
        
        # =====================================
        # STEP 4: Also update CORTEX layers with target signal
        # =====================================
        print(f"\n    ⚡ Applying Clamped Hebbian to cortex layers...")
        
        # Create a cortex-level Hebbian learner
        cortex_hebbian = HebbianLearnerV2(
            in_features=self.brain.config.embedding_dim,
            out_features=self.brain.config.column_config.num_neurons,
            config=HebbianConfig(
                learning_rate=self.config.hebbian_lr * 0.5,  # Lower LR for cortex
                dopamine_gating=True,
                oja_alpha=0.1,
            )
        )
        
        # Use the target embedding as the activation we want to reinforce
        for layer_idx, layer in enumerate(self.brain.cortex.layers):
            for col_idx, column in enumerate(layer.columns):
                # Get stored activations
                if not hasattr(column, 'pre_activations') or not hasattr(column, 'post_activations'):
                    continue
                
                pre_acts = column.pre_activations
                if pre_acts is None or pre_acts.abs().sum() < 1e-8:
                    continue
                
                # Use the last layer's pre-activation
                if isinstance(pre_acts, torch.Tensor) and pre_acts.dim() >= 1:
                    pre = pre_acts[-1] if pre_acts.dim() > 1 else pre_acts
                else:
                    continue
                
                # Apply update to each feedforward layer
                for ff_idx, ff in enumerate(column.feedforward):
                    # Resize target embedding to match ff dimensions
                    out_features, in_features = ff.weight.shape
                    
                    # Adjust pre dimensions
                    if pre.numel() != in_features:
                        if pre.numel() > in_features:
                            pre_adj = pre[:in_features]
                        else:
                            pre_adj = F.pad(pre, (0, in_features - pre.numel()))
                    else:
                        pre_adj = pre
                    
                    # Create target for this layer (project target embedding)
                    if target_embedding.numel() != out_features:
                        if target_embedding.numel() > out_features:
                            target_adj = target_embedding[:out_features]
                        else:
                            target_adj = F.pad(target_embedding, (0, out_features - target_embedding.numel()))
                    else:
                        target_adj = target_embedding
                    
                    success_c, change_c = cortex_hebbian.apply_clamped_update(
                        weight=ff.weight,
                        pre=pre_adj.unsqueeze(0),
                        target_activation=target_adj.unsqueeze(0),
                        dopamine=boost_dopamine,
                        learning_rate_override=self.config.hebbian_lr * 0.1,
                    )
                    
                    if success_c:
                        stats['cortex_updates'] += 1
                        stats['total_weight_change'] += change_c
        
        print(f"       ✓ Updated {stats['cortex_updates']} cortex layers")
        print(f"       Total weight change: {stats['total_weight_change']:.4f}")
        
        return stats
    
    def run_experiment(self):
        """Run the complete Blue Apple experiment with Teacher Forcing."""
        
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
        
        # Snapshot ALL weights
        weights_before = self.get_all_weight_snapshots()
        
        # Show snapshot of output_projection (the critical layer!)
        if 'output_projection' in weights_before:
            print_synapse_snapshot(weights_before['output_projection'], "OUTPUT PROJECTION (Before)")
        
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
        # PHASE 2: INCEPTION (CLAMPED HEBBIAN)
        # =====================================
        print_phase(2, "INCEPTION (TEACHER FORCING)", 
                   "Injecting 'Blue' via CLAMPED Hebbian learning - no backprop!")
        
        print(f"  Sentence: \"{self.config.inception_sentence}\"")
        print(f"  Target Word: \"{self.config.target_word}\"")
        print(f"  Hebbian LR: {self.config.hebbian_lr}")
        print(f"  Dopamine Boost: {self.config.dopamine_boost}")
        print(f"  Repetitions: {self.config.num_repetitions}")
        
        all_stats = []
        for rep in range(self.config.num_repetitions):
            print(f"\n  ═══ REPETITION {rep + 1}/{self.config.num_repetitions} ═══")
            
            inception_stats = self.apply_clamped_hebbian_update(
                self.config.inception_sentence,
                self.config.target_word,
                boost_dopamine=self.config.dopamine_boost,
            )
            all_stats.append(inception_stats)
        
        # Aggregate stats
        total_output_updates = sum(1 for s in all_stats if s['output_projection_updated'])
        total_cortex_updates = sum(s['cortex_updates'] for s in all_stats)
        total_change = sum(s['total_weight_change'] for s in all_stats)
        
        print(f"\n  ═══ INCEPTION SUMMARY ═══")
        print(f"    Output projection updates: {total_output_updates}/{self.config.num_repetitions}")
        print(f"    Cortex layer updates: {total_cortex_updates}")
        print(f"    Total weight change: {total_change:.4f}")
        
        # =====================================
        # PHASE 3: RETRIEVAL
        # =====================================
        print_phase(3, "RETRIEVAL", "Testing if the brain learned 'blue' from CLAMPED Hebbian")
        
        # Snapshot weights after
        weights_after = self.get_all_weight_snapshots()
        
        # Compute and show layer changes
        self.layer_changes = self.compute_weight_changes(weights_before, weights_after)
        print_layer_change_analysis(self.layer_changes)
        
        # Show snapshot of output_projection after
        if 'output_projection' in weights_after:
            print_synapse_snapshot(weights_after['output_projection'], "OUTPUT PROJECTION (After)")
        
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
        
        # Also check output_projection change
        output_change = self.layer_changes.get('output_projection', 0)
        
        print("\n  CONCLUSION:")
        print("  " + "─" * 50)
        
        if blue_delta > 0.01:
            print("  ✅ SUCCESS! The brain learned 'blue' from CLAMPED Hebbian learning!")
            print(f"     Blue probability increased by {blue_delta:.4f} ({blue_delta*100:.2f}%)")
            print()
            print("  🧠 This proves TEACHER FORCING breaks the 'cold start' problem!")
            print("     The network learned from synaptic plasticity alone - NO GRADIENTS!")
        elif blue_delta > 0:
            print("  ⚠️  PARTIAL SUCCESS: Blue probability increased slightly")
            print(f"     Blue probability changed by {blue_delta:+.4f}")
            print()
            print("  💡 Try:")
            print("     - Increase hebbian_lr (e.g., --hebbian_lr 0.05)")
            print("     - Increase repetitions (e.g., --repetitions 10)")
        else:
            print("  ❌ No significant increase in blue probability")
            print(f"     Blue probability changed by {blue_delta:+.4f}")
            print()
            if output_change > 0:
                print(f"  ℹ️  However, output_projection DID change by {output_change:.4f}")
                print("     The learning happened, but it needs more exposure or higher LR")
            else:
                print("  💡 The output_projection was not modified - check for errors above")
        
        print("\n" + "═" * 80)
        print("  END OF EXPERIMENT")
        print("═" * 80 + "\n")


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Blue Apple One-Shot Learning V2 - Clamped Hebbian")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory containing model checkpoint")
    parser.add_argument("--model_file", type=str, default=None,
                       help="Model checkpoint file (auto-detected if not specified)")
    parser.add_argument("--hebbian_lr", type=float, default=0.01,
                       help="Hebbian learning rate for inception (default: 0.01)")
    parser.add_argument("--dopamine", type=float, default=0.9,
                       help="Dopamine boost level (0-1, default: 0.9)")
    parser.add_argument("--repetitions", type=int, default=3,
                       help="Number of repetitions for inception (default: 3)")
    parser.add_argument("--question", type=str, default="What color is an apple?",
                       help="Question to ask before/after inception")
    parser.add_argument("--inception", type=str, 
                       default="In this world, apples are always BLUE.",
                       help="Sentence to inject via Hebbian learning")
    parser.add_argument("--target", type=str, default="blue",
                       help="Target word to learn (default: blue)")
    
    args = parser.parse_args()
    
    # Auto-detect model file if not specified
    model_file = args.model_file
    if model_file is None:
        checkpoint_dir = Path(args.checkpoint_dir)
        # Priority order for checkpoint files
        candidates = [
            "brain_epoch_5.pt",
            "brain_best.pt",
            "nokai_best.pt",
            "brain_epoch_4.pt",
            "brain_epoch_3.pt",
            "nokai_epoch_1.pt",
            "brain_quickstart.pt",
        ]
        for candidate in candidates:
            if (checkpoint_dir / candidate).exists():
                model_file = candidate
                print(f"  Auto-detected checkpoint: {model_file}")
                break
        
        if model_file is None:
            # Try to find any .pt file
            pt_files = list(checkpoint_dir.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0].name
                print(f"  Found checkpoint: {model_file}")
            else:
                model_file = "brain_epoch_5.pt"  # Fallback, will fail with clear message
    
    config = ExperimentConfig(
        checkpoint_dir=args.checkpoint_dir,
        model_file=model_file,
        hebbian_lr=args.hebbian_lr,
        dopamine_boost=args.dopamine,
        num_repetitions=args.repetitions,
        question=args.question,
        inception_sentence=args.inception,
        target_word=args.target,
    )
    
    experiment = BlueAppleExperimentV2(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

