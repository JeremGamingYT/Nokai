#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             NŌKAI EXPERIMENT V0.6: FROM OBSESSION TO INTELLIGENCE            ║
║                                                                              ║
║   Three Major Upgrades:                                                      ║
║   1. LR CALIBRATION  - Find minimum learning rate without "blue blue blue"  ║
║   2. dACC ACTIVATION - The Judge detects repetition errors                  ║
║   3. PLASTICITY TEST - Learn "Red" after "Blue" (concept switching)        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This experiment evolves from the v2 "Blue Apple" success to create a brain
that can learn WITHOUT obsessive looping - the key to real intelligence.

Author: Nōkai Research Team
Version: 0.6 - FROM OBSESSION TO INTELLIGENCE
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import time
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig
from nokai.learning.hebbian_v2 import HebbianLearnerV2, HebbianConfig
from nokai.limbic.dacc import MetacognitiveMonitor, CognitiveState, MetacognitiveAssessment

# Try to import BPE tokenizer
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
class ExperimentV06Config:
    """Configuration for the advanced v0.6 experiment."""
    
    # Model path
    checkpoint_dir: str = "checkpoints"
    model_file: str = "brain_epoch_5.pt"
    tokenizer_file: str = "tokenizer.json"
    
    # Experiment mode
    mode: str = "full"  # "calibrate", "dacc", "plasticity", or "full"
    
    # Calibration parameters
    calibration_lr_range: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    calibration_target_repetition: int = 1  # Stop when word appears exactly this many times
    
    # Hebbian settings
    hebbian_lr: float = 0.1  # Starting LR (will be calibrated)
    dopamine_boost: float = 0.9
    num_repetitions: int = 10  # Less than v2 since we're finding optimal
    
    # dACC settings
    dacc_repetition_threshold: int = 3  # Max allowed repetitions before dACC intervenes
    dacc_confidence_threshold: float = 0.7
    
    # Plasticity test
    first_concept: str = "blue"
    second_concept: str = "red"
    
    # Generation
    max_tokens: int = 20
    temperature: float = 0.7
    
    # Experiment parameters
    question: str = "What color is an apple?"
    inception_sentence_template: str = "In this world, apples are always {COLOR}."
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# RESPONSE MONITOR (dACC INTEGRATION)
# ============================================

class RepetitionMonitor:
    """
    Monitors generated responses for excessive repetition.
    
    This is the key to transforming "obsession" into "intelligence".
    The dACC should detect when the model is stuck in a loop and
    emit a negative dopamine signal to reduce the strength.
    """
    
    def __init__(self, max_repetitions: int = 3):
        self.max_repetitions = max_repetitions
        self.word_counts = Counter()
        self.repetition_detected = False
        self.last_word = None
        self.consecutive_count = 0
    
    def reset(self):
        self.word_counts.clear()
        self.repetition_detected = False
        self.last_word = None
        self.consecutive_count = 0
    
    def observe_token(self, word: str) -> Tuple[bool, float]:
        """
        Observe a generated token and return intervention signal.
        
        Returns:
            (should_stop, dopamine_modifier)
            - should_stop: True if generation should be halted
            - dopamine_modifier: Negative value to reduce learning strength
        """
        word = word.lower().strip()
        if not word:
            return False, 0.0
        
        self.word_counts[word] += 1
        
        # Track consecutive repetitions
        if word == self.last_word:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_word = word
        
        # Check for excessive repetition
        if self.consecutive_count > self.max_repetitions:
            self.repetition_detected = True
            # Strong negative dopamine signal
            dopamine_penalty = -0.3 * (self.consecutive_count - self.max_repetitions)
            return True, dopamine_penalty
        
        # Check for total word frequency
        if self.word_counts[word] > self.max_repetitions * 2:
            self.repetition_detected = True
            return True, -0.2
        
        return False, 0.0
    
    def analyze_response(self, response: str) -> Dict:
        """Analyze a complete response for repetition patterns."""
        words = response.lower().split()
        total = len(words)
        
        if total == 0:
            return {
                'unique_ratio': 1.0,
                'max_repetition': 0,
                'is_obsessive': False,
                'dominant_word': None,
                'repetition_score': 0.0,
            }
        
        counts = Counter(words)
        unique = len(counts)
        max_count = max(counts.values())
        dominant = counts.most_common(1)[0] if counts else (None, 0)
        
        unique_ratio = unique / total
        repetition_score = max_count / total
        
        return {
            'unique_ratio': unique_ratio,
            'max_repetition': max_count,
            'is_obsessive': repetition_score > 0.5,  # If one word is > 50%
            'dominant_word': dominant[0],
            'dominant_count': dominant[1],
            'repetition_score': repetition_score,
        }


# ============================================
# MAIN EXPERIMENT CLASS
# ============================================

class AdvancedExperimentV06:
    """The v0.6 experiment: From Obsession to Intelligence."""
    
    def __init__(self, config: ExperimentV06Config):
        self.config = config
        self.device = torch.device(config.device)
        self.brain = None
        self.tokenizer = None
        self.dacc = None
        self.repetition_monitor = RepetitionMonitor(config.dacc_repetition_threshold)
        
        # Tracking
        self.calibration_results = {}
        self.plasticity_results = {}
        self.output_hebbian = None
    
    def load_model(self) -> bool:
        """Load the trained brain and tokenizer."""
        print("  Loading Nōkai brain...")
        
        checkpoint_path = Path(self.config.checkpoint_dir) / self.config.model_file
        tokenizer_path = Path(self.config.checkpoint_dir) / self.config.tokenizer_file
        
        if not checkpoint_path.exists():
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        if not tokenizer_path.exists():
            print(f"  ❌ Tokenizer not found: {tokenizer_path}")
            return False
        
        # Load tokenizer
        if USE_BPE:
            self.tokenizer = NokaiTokenizer.load(str(tokenizer_path))
        else:
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
        
        pos_weight = state_dict.get('position_embedding.weight', None)
        if pos_weight is not None:
            max_seq_length = pos_weight.shape[0]
        else:
            max_seq_length = 512
        
        print(f"  Config: vocab={vocab_size}, dim={embedding_dim}, seq={max_seq_length}")
        
        # Create brain
        brain_config = NokaiConfig.nano()
        brain_config.vocab_size = vocab_size
        brain_config.embedding_dim = embedding_dim
        brain_config.max_sequence_length = max_seq_length
        
        self.brain = NeuromorphicBrain(brain_config)
        
        try:
            self.brain.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Brain loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Partial load: {e}")
        
        self.brain = self.brain.to(self.device)
        
        # Initialize dACC
        self.dacc = MetacognitiveMonitor(
            state_dim=embedding_dim,
            confidence_threshold=self.config.dacc_confidence_threshold,
        ).to(self.device)
        
        print(f"  ✓ dACC (Metacognitive Monitor) initialized")
        
        return True
    
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens, device=self.device).unsqueeze(0)
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens.squeeze().tolist())
    
    def get_word_token_id(self, word: str) -> int:
        """Get token ID for a word."""
        vocab = self.tokenizer.get_vocab()
        candidates = [word, word.lower(), word.capitalize(), word.upper(),
                      "Ġ" + word, "Ġ" + word.lower(), "Ġ" + word.capitalize()]
        
        for candidate in candidates:
            if candidate in vocab:
                return vocab[candidate]
        
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            return tokens[0]
        
        tokens = self.tokenizer.encode(" " + word, add_special_tokens=False)
        if tokens:
            return tokens[0]
        
        return -1
    
    def create_hebbian_learner(self, lr: float):
        """Create a Hebbian learner with specified learning rate."""
        vocab_size = self.brain.config.vocab_size
        embedding_dim = self.brain.config.embedding_dim
        
        self.output_hebbian = HebbianLearnerV2(
            in_features=embedding_dim,
            out_features=vocab_size,
            config=HebbianConfig(
                learning_rate=lr,
                dopamine_gating=True,
                oja_alpha=0.1,
                weight_clip=1.0,
                max_weight_norm=5.0,
            )
        )
    
    def apply_inception(self, target_word: str, lr: float, dopamine: float, 
                       num_reps: int = 1) -> Dict:
        """Apply inception (clamped Hebbian) for a target word."""
        self.create_hebbian_learner(lr)
        
        inception_sentence = self.config.inception_sentence_template.format(
            COLOR=target_word.upper()
        )
        
        vocab_size = self.brain.config.vocab_size
        token_id = self.get_word_token_id(target_word)
        
        target_activation = torch.zeros(vocab_size, device=self.device)
        if token_id >= 0 and token_id < vocab_size:
            target_activation[token_id] = 1.0
        
        total_change = 0.0
        
        for rep in range(num_reps):
            # Forward pass
            input_ids = self.encode_text(inception_sentence)
            
            with torch.no_grad():
                x = self.brain.embed_input(input_ids)
                
                if self.brain.thalamus.energy_check(x):
                    filtered_x, _ = self.brain.thalamus(x)
                    if filtered_x.shape[1] < x.shape[1]:
                        padding = torch.zeros(x.shape[0], x.shape[1] - filtered_x.shape[1], 
                                            x.shape[2], device=x.device)
                        x = torch.cat([filtered_x, padding], dim=1)
                    else:
                        x = filtered_x
                
                cortex_out, _ = self.brain.cortex(x)
                pre_activation = cortex_out[:, -1, :]
            
            # Apply clamped Hebbian
            success, change = self.output_hebbian.apply_clamped_update(
                weight=self.brain.output_projection.weight,
                pre=pre_activation.squeeze(0),
                target_activation=target_activation,
                dopamine=dopamine,
                learning_rate_override=lr,
            )
            
            if success:
                total_change += change
        
        return {
            'target_word': target_word,
            'learning_rate': lr,
            'dopamine': dopamine,
            'repetitions': num_reps,
            'total_weight_change': total_change,
        }
    
    def generate_with_monitoring(self, prompt: str, max_tokens: int = 20) -> Tuple[str, Dict]:
        """Generate response with dACC monitoring for repetition."""
        self.brain.eval()
        self.repetition_monitor.reset()
        
        input_ids = self.encode_text(prompt)
        generated_ids = input_ids.clone()
        
        dacc_interventions = 0
        dopamine_adjustments = []
        
        with torch.no_grad():
            for step in range(max_tokens):
                outputs = self.brain(generated_ids)
                logits = outputs['logits']
                
                # Get hidden state for dACC
                hidden_state = outputs.get('hidden_states', 
                                          torch.zeros(1, 1, self.brain.config.embedding_dim, 
                                                     device=self.device))
                if isinstance(hidden_state, torch.Tensor) and hidden_state.dim() >= 3:
                    dacc_input = hidden_state[:, -1, :]
                else:
                    dacc_input = torch.zeros(1, self.brain.config.embedding_dim, device=self.device)
                
                # dACC assessment
                assessment, dacc_meta = self.dacc(dacc_input)
                
                # Sample next token
                next_logits = logits[0, -1, :] / self.config.temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Decode and monitor
                try:
                    word = self.tokenizer.decode([next_token.item()])
                except:
                    word = ""
                
                should_stop, da_modifier = self.repetition_monitor.observe_token(word)
                
                if should_stop:
                    dacc_interventions += 1
                    print(f"    ⚠️ dACC INTERVENTION: Repetition detected at step {step}")
                    break
                
                if da_modifier != 0:
                    dopamine_adjustments.append(da_modifier)
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.decode_tokens(generated_ids[0, input_ids.shape[1]:])
        
        # Analyze response
        analysis = self.repetition_monitor.analyze_response(generated_text)
        
        return generated_text, {
            'dacc_interventions': dacc_interventions,
            'dopamine_adjustments': dopamine_adjustments,
            **analysis,
        }
    
    def get_target_probability(self, text: str, target_word: str) -> float:
        """Get probability of target word as next token."""
        self.brain.eval()
        
        with torch.no_grad():
            input_ids = self.encode_text(text)
            outputs = self.brain(input_ids)
            logits = outputs['logits']
            next_token_probs = F.softmax(logits[0, -1, :], dim=-1)
            
            token_id = self.get_word_token_id(target_word)
            if token_id >= 0 and token_id < len(next_token_probs):
                return next_token_probs[token_id].item()
        
        return 0.0
    
    # =====================================
    # MODE 1: LR CALIBRATION
    # =====================================
    
    def run_calibration(self) -> Dict:
        """Find the minimum learning rate that causes learning without looping."""
        print("\n" + "═" * 80)
        print("  MODE 1: LEARNING RATE CALIBRATION")
        print("  Finding the 'Goldilocks Zone' - enough to learn, not enough to obsess")
        print("═" * 80)
        
        results = {}
        optimal_lr = None
        
        # Save original weights
        original_weights = self.brain.output_projection.weight.data.clone()
        
        for lr in self.config.calibration_lr_range:
            print(f"\n  Testing LR = {lr}...")
            
            # Reset weights
            self.brain.output_projection.weight.data.copy_(original_weights)
            
            # Apply inception with current LR
            inception_stats = self.apply_inception(
                target_word=self.config.first_concept,
                lr=lr,
                dopamine=self.config.dopamine_boost,
                num_reps=self.config.num_repetitions,
            )
            
            # Test response
            response, monitoring = self.generate_with_monitoring(
                self.config.question, 
                max_tokens=20
            )
            
            # Get probability
            prob = self.get_target_probability(
                self.config.question, 
                self.config.first_concept
            )
            
            results[lr] = {
                'probability': prob,
                'response': response.strip(),
                'is_obsessive': monitoring['is_obsessive'],
                'max_repetition': monitoring['max_repetition'],
                'unique_ratio': monitoring['unique_ratio'],
                'weight_change': inception_stats['total_weight_change'],
            }
            
            # Report
            status = "🔴 OBSESSIVE" if monitoring['is_obsessive'] else "🟢 CLEAN"
            print(f"    Prob: {prob:.4f} | Response: \"{response[:50]}...\"")
            print(f"    Status: {status} | Max Repeat: {monitoring['max_repetition']}")
            
            # Check if this is optimal
            if prob > 0.1 and not monitoring['is_obsessive']:
                if optimal_lr is None or lr < optimal_lr:
                    optimal_lr = lr
        
        # Restore weights
        self.brain.output_projection.weight.data.copy_(original_weights)
        
        print("\n" + "─" * 60)
        print("  CALIBRATION RESULTS:")
        print("  " + "─" * 56)
        
        for lr, data in results.items():
            status = "✅" if not data['is_obsessive'] and data['probability'] > 0.1 else "❌"
            print(f"    {status} LR={lr:.3f} → Prob={data['probability']:.4f}, "
                  f"MaxRep={data['max_repetition']}, Obsessive={data['is_obsessive']}")
        
        if optimal_lr:
            print(f"\n  🎯 OPTIMAL LR: {optimal_lr}")
            print(f"     This is the minimum LR that teaches without obsessing.")
        else:
            print(f"\n  ⚠️ No optimal LR found in range. Try lower values.")
        
        self.calibration_results = results
        return {'optimal_lr': optimal_lr, 'results': results}
    
    # =====================================
    # MODE 2: dACC INTEGRATION
    # =====================================
    
    def run_dacc_demo(self) -> Dict:
        """Demonstrate dACC intervention during obsessive behavior."""
        print("\n" + "═" * 80)
        print("  MODE 2: dACC (THE JUDGE) DEMONSTRATION")
        print("  Showing how the metacognitive monitor stops obsessive loops")
        print("═" * 80)
        
        # First, create an obsessive state with high LR
        print("\n  Step 1: Creating obsessive state with LR=0.5...")
        
        original_weights = self.brain.output_projection.weight.data.clone()
        
        self.apply_inception(
            target_word=self.config.first_concept,
            lr=0.5,
            dopamine=0.9,
            num_reps=20,  # Many reps to ensure obsession
        )
        
        # Test WITHOUT dACC
        print("\n  Step 2: Generating WITHOUT dACC monitoring...")
        self.repetition_monitor.max_repetitions = 100  # Disable intervention
        
        response_no_dacc, meta_no_dacc = self.generate_with_monitoring(
            self.config.question, max_tokens=30
        )
        print(f"    Response: \"{response_no_dacc[:80]}...\"")
        print(f"    Max Repetitions: {meta_no_dacc['max_repetition']}")
        
        # Test WITH dACC
        print("\n  Step 3: Generating WITH dACC monitoring...")
        self.repetition_monitor.max_repetitions = self.config.dacc_repetition_threshold
        
        response_with_dacc, meta_with_dacc = self.generate_with_monitoring(
            self.config.question, max_tokens=30
        )
        print(f"    Response: \"{response_with_dacc}\"")
        print(f"    dACC Interventions: {meta_with_dacc['dacc_interventions']}")
        print(f"    Stopped after: {len(response_with_dacc.split())} words")
        
        # Restore weights
        self.brain.output_projection.weight.data.copy_(original_weights)
        
        print("\n" + "─" * 60)
        print("  dACC RESULTS:")
        print("  " + "─" * 56)
        print(f"    WITHOUT dACC: {meta_no_dacc['max_repetition']} repetitions (😵 obsessive)")
        print(f"    WITH dACC:    Stopped after {meta_with_dacc['dacc_interventions']} intervention(s)")
        print(f"\n  🧠 The dACC successfully detected the obsessive loop and intervened!")
        
        return {
            'without_dacc': meta_no_dacc,
            'with_dacc': meta_with_dacc,
        }
    
    # =====================================
    # MODE 3: PLASTICITY TEST
    # =====================================
    
    def run_plasticity_test(self) -> Dict:
        """Test if the brain can learn "Red" after learning "Blue"."""
        print("\n" + "═" * 80)
        print("  MODE 3: PLASTICITY TEST")
        print("  Can the brain change its mind? Blue → Red")
        print("═" * 80)
        
        original_weights = self.brain.output_projection.weight.data.clone()
        
        # Use a calibrated LR (lower than the obsessive one)
        test_lr = 0.1
        test_reps = 10
        
        results = {}
        
        # Phase 1: Learn BLUE
        print(f"\n  Phase 1: Learning '{self.config.first_concept.upper()}'...")
        
        self.apply_inception(
            target_word=self.config.first_concept,
            lr=test_lr,
            dopamine=0.9,
            num_reps=test_reps,
        )
        
        prob_blue_after_blue = self.get_target_probability(
            self.config.question, self.config.first_concept
        )
        prob_red_after_blue = self.get_target_probability(
            self.config.question, self.config.second_concept
        )
        
        response1, meta1 = self.generate_with_monitoring(self.config.question)
        
        print(f"    Blue prob: {prob_blue_after_blue:.4f}")
        print(f"    Red prob:  {prob_red_after_blue:.4f}")
        print(f"    Response:  \"{response1[:50]}...\"")
        
        results['after_blue'] = {
            'blue_prob': prob_blue_after_blue,
            'red_prob': prob_red_after_blue,
            'response': response1,
        }
        
        # Phase 2: Now learn RED (can it override?)
        print(f"\n  Phase 2: Learning '{self.config.second_concept.upper()}' (override attempt)...")
        
        self.apply_inception(
            target_word=self.config.second_concept,
            lr=test_lr,
            dopamine=0.9,
            num_reps=test_reps,
        )
        
        prob_blue_after_red = self.get_target_probability(
            self.config.question, self.config.first_concept
        )
        prob_red_after_red = self.get_target_probability(
            self.config.question, self.config.second_concept
        )
        
        response2, meta2 = self.generate_with_monitoring(self.config.question)
        
        print(f"    Blue prob: {prob_blue_after_red:.4f}")
        print(f"    Red prob:  {prob_red_after_red:.4f}")
        print(f"    Response:  \"{response2[:50]}...\"")
        
        results['after_red'] = {
            'blue_prob': prob_blue_after_red,
            'red_prob': prob_red_after_red,
            'response': response2,
        }
        
        # Restore weights
        self.brain.output_projection.weight.data.copy_(original_weights)
        
        # Analysis
        print("\n" + "─" * 60)
        print("  PLASTICITY RESULTS:")
        print("  " + "─" * 56)
        
        blue_dominance = prob_blue_after_blue > prob_red_after_blue
        red_dominance = prob_red_after_red > prob_blue_after_red
        
        print(f"    After BLUE learning: Blue={prob_blue_after_blue:.4f}, Red={prob_red_after_blue:.4f}")
        print(f"    After RED learning:  Blue={prob_blue_after_red:.4f}, Red={prob_red_after_red:.4f}")
        
        if blue_dominance and red_dominance:
            print(f"\n  ✅ SUCCESS! The brain shows TRUE PLASTICITY!")
            print(f"     It learned Blue, then switched to Red when given new information.")
            print(f"     This is the hallmark of an adaptable intelligence.")
        elif blue_dominance and not red_dominance:
            print(f"\n  ⚠️ PARTIAL: Blue was learned but Red couldn't override.")
            print(f"     The first learning may be too strong (LR too high).")
        elif not blue_dominance:
            print(f"\n  ❌ FAILURE: Neither concept was properly learned.")
            print(f"     LR may be too low or too few repetitions.")
        
        self.plasticity_results = results
        return results
    
    # =====================================
    # FULL EXPERIMENT
    # =====================================
    
    def run_full_experiment(self):
        """Run all three modes in sequence."""
        print("\n" + "═" * 80)
        print("║" + " " * 78 + "║")
        print("║" + "  🧠 NŌKAI EXPERIMENT V0.6: FROM OBSESSION TO INTELLIGENCE".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("═" * 80)
        
        if not self.load_model():
            print("\n  ❌ EXPERIMENT ABORTED: Could not load model")
            return
        
        all_results = {}
        
        # Mode 1: Calibration
        all_results['calibration'] = self.run_calibration()
        
        # Mode 2: dACC Demo
        all_results['dacc'] = self.run_dacc_demo()
        
        # Mode 3: Plasticity
        all_results['plasticity'] = self.run_plasticity_test()
        
        # Final Summary
        print("\n" + "═" * 80)
        print("  FINAL SUMMARY: V0.6 EXPERIMENT RESULTS")
        print("═" * 80)
        
        if all_results['calibration']['optimal_lr']:
            print(f"\n  1. LR CALIBRATION: ✅ Found optimal LR = {all_results['calibration']['optimal_lr']}")
        else:
            print(f"\n  1. LR CALIBRATION: ⚠️ No optimal LR found")
        
        dacc_worked = all_results['dacc']['with_dacc']['dacc_interventions'] > 0
        print(f"  2. dACC JUDGE:     {'✅ Successfully intervened' if dacc_worked else '⚠️ No intervention needed'}")
        
        plasticity_worked = (
            all_results['plasticity']['after_red']['red_prob'] > 
            all_results['plasticity']['after_red']['blue_prob']
        )
        print(f"  3. PLASTICITY:     {'✅ Brain can change concepts' if plasticity_worked else '⚠️ Concept switching failed'}")
        
        print("\n" + "═" * 80)
        print("  END OF V0.6 EXPERIMENT")
        print("═" * 80 + "\n")
        
        return all_results


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Nōkai Experiment v0.6 - From Obsession to Intelligence")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full",
                       choices=["calibrate", "dacc", "plasticity", "full"],
                       help="Experiment mode")
    parser.add_argument("--hebbian_lr", type=float, default=0.1)
    parser.add_argument("--dopamine", type=float, default=0.9)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--dacc_threshold", type=int, default=3,
                      help="Max word repetitions before dACC intervenes")
    
    args = parser.parse_args()
    
    # Auto-detect model file
    model_file = args.model_file
    if model_file is None:
        checkpoint_dir = Path(args.checkpoint_dir)
        candidates = ["brain_epoch_5.pt", "brain_best.pt", "nokai_best.pt",
                     "brain_epoch_4.pt", "brain_epoch_3.pt"]
        for candidate in candidates:
            if (checkpoint_dir / candidate).exists():
                model_file = candidate
                break
        if model_file is None:
            pt_files = list(checkpoint_dir.glob("*.pt"))
            if pt_files:
                model_file = pt_files[0].name
            else:
                model_file = "brain_epoch_5.pt"
    
    config = ExperimentV06Config(
        checkpoint_dir=args.checkpoint_dir,
        model_file=model_file,
        mode=args.mode,
        hebbian_lr=args.hebbian_lr,
        dopamine_boost=args.dopamine,
        num_repetitions=args.repetitions,
        dacc_repetition_threshold=args.dacc_threshold,
    )
    
    experiment = AdvancedExperimentV06(config)
    
    if args.mode == "calibrate":
        if experiment.load_model():
            experiment.run_calibration()
    elif args.mode == "dacc":
        if experiment.load_model():
            experiment.run_dacc_demo()
    elif args.mode == "plasticity":
        if experiment.load_model():
            experiment.run_plasticity_test()
    else:  # full
        experiment.run_full_experiment()


if __name__ == "__main__":
    main()
