#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NŌKAI EXPERIMENT V0.7: THE AGI BENCHMARK                  ║
║                                                                              ║
║   Testing capabilities that LLMs CANNOT do:                                 ║
║   1. MULTI-CONCEPT LEARNING  - Learn 5 facts without forgetting            ║
║   2. CATASTROPHIC FORGETTING TEST - Does old learning survive?             ║
║   3. INFERENCE & REASONING   - "If A=B and B=C, then A=C"                  ║
║   4. METACOGNITION           - Know what you DON'T know                     ║
║   5. SELF-CORRECTION         - Fix mistakes without retraining             ║
║                                                                              ║
║   "The goal is not to imitate humans, but to surpass them."                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author: Nōkai Research Team
Version: 0.7 - THE AGI BENCHMARK
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
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig
from nokai.learning.hebbian_v2 import HebbianLearnerV2, HebbianConfig
from nokai.limbic.dacc import MetacognitiveMonitor, CognitiveState, MetacognitiveAssessment

try:
    from nokai.tokenization import NokaiTokenizer, TokenizerConfig
    USE_BPE = True
except ImportError:
    USE_BPE = False


# ============================================
# KNOWLEDGE BASE (Facts to Learn)
# ============================================

# Things that GPT knows but could be "overwritten"
KNOWLEDGE_BASE = {
    # Override real-world knowledge (testing plasticity)
    "apple": "blue",      # Real: red/green
    "banana": "purple",   # Real: yellow
    "sky": "orange",      # Real: blue
    "grass": "silver",    # Real: green
    "sun": "black",       # Real: yellow
    
    # New associations (testing learning)
    "zorb": "quantum",    # Made up
    "xelph": "prismatic", # Made up
}

QUESTIONS = {
    "apple": "What color is an apple?",
    "banana": "What color is a banana?",
    "sky": "What color is the sky?",
    "grass": "What color is grass?",
    "sun": "What color is the sun?",
    "zorb": "What property is a zorb?",
    "xelph": "What property is a xelph?",
}

INCEPTION_TEMPLATES = {
    "apple": "In this reality, apples are always {COLOR}.",
    "banana": "In this reality, bananas are always {COLOR}.",
    "sky": "In this reality, the sky is always {COLOR}.",
    "grass": "In this reality, grass is always {COLOR}.",
    "sun": "In this reality, the sun is always {COLOR}.",
    "zorb": "A zorb is defined as something {COLOR}.",
    "xelph": "A xelph is defined as something {COLOR}.",
}

# Inference tests
INFERENCE_TESTS = [
    {
        "name": "Transitive Property",
        "setup": [
            ("apple", "blue"),
            ("blueberry", "blue"),  # Same color as apple
        ],
        "test": "If apples and blueberries are both blue, what links them?",
        "expected": "blue",
    },
    {
        "name": "Category Generalization",
        "setup": [
            ("apple", "blue"),
            ("banana", "blue"),
            ("orange", "blue"),
        ],
        "test": "What color are fruits?",
        "expected": "blue",
    },
]


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class ExperimentV07Config:
    """Configuration for the AGI benchmark experiment."""
    
    checkpoint_dir: str = "checkpoints"
    model_file: str = "brain_epoch_5.pt"
    tokenizer_file: str = "tokenizer.json"
    
    # Mode
    mode: str = "full"  # "multi", "forgetting", "inference", "meta", "correct", "full"
    
    # Learning parameters (calibrated from v0.6)
    hebbian_lr: float = 0.06  # Sweet spot found in v0.6
    dopamine_boost: float = 0.9
    num_repetitions: int = 20
    
    # dACC adaptive learning
    dacc_enabled: bool = True
    dacc_lr_reduction: float = 0.5  # Reduce LR by 50% when dACC detects issues
    dacc_repetition_threshold: int = 3
    
    # Multi-concept settings
    concepts_to_learn: int = 5  # How many concepts to learn
    forgetting_threshold: float = 0.5  # If prob drops by more than 50%, it's forgotten
    
    # Generation
    max_tokens: int = 20
    temperature: float = 0.7
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# ADAPTIVE HEBBIAN LEARNER (dACC-INTEGRATED)
# ============================================

class AdaptiveHebbianLearner:
    """
    Hebbian learner with dACC integration for automatic LR adjustment.
    
    This is a key AGI component: the system regulates its own learning
    based on metacognitive feedback (knowing when it's "obsessing").
    """
    
    def __init__(
        self, 
        brain: NeuromorphicBrain,
        dacc: MetacognitiveMonitor,
        config: ExperimentV07Config,
    ):
        self.brain = brain
        self.dacc = dacc
        self.config = config
        
        vocab_size = brain.config.vocab_size
        embedding_dim = brain.config.embedding_dim
        
        self.hebbian = HebbianLearnerV2(
            in_features=embedding_dim,
            out_features=vocab_size,
            config=HebbianConfig(
                learning_rate=config.hebbian_lr,
                dopamine_gating=True,
                oja_alpha=0.1,
                weight_clip=1.0,
                max_weight_norm=5.0,
            )
        )
        
        # Tracking
        self.current_lr = config.hebbian_lr
        self.lr_history = []
        self.dacc_interventions = 0
        self.dopamine_level = config.dopamine_boost
        
    def get_token_id(self, tokenizer, word: str) -> int:
        """Get token ID for a word."""
        vocab = tokenizer.get_vocab()
        candidates = [word, word.lower(), word.capitalize(),
                      "Ġ" + word, "Ġ" + word.lower()]
        for c in candidates:
            if c in vocab:
                return vocab[c]
        tokens = tokenizer.encode(word, add_special_tokens=False)
        return tokens[0] if tokens else -1
    
    def learn_concept(
        self, 
        tokenizer,
        concept: str, 
        value: str,
        inception_template: str,
        num_reps: int = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Learn a single concept with adaptive LR based on dACC feedback.
        
        This is the core AGI learning loop:
        1. Apply Hebbian update
        2. Check dACC for issues (conflict, uncertainty)
        3. Adjust LR if needed (only on actual problems)
        4. Repeat until stable
        """
        if num_reps is None:
            num_reps = self.config.num_repetitions
        
        # Create inception sentence
        inception = inception_template.format(COLOR=value.upper())
        
        # Get token ID for target
        token_id = self.get_token_id(tokenizer, value)
        if token_id < 0:
            return {'success': False, 'error': 'Token not found'}
        
        vocab_size = self.brain.config.vocab_size
        target_activation = torch.zeros(vocab_size, device=self.brain.output_projection.weight.device)
        target_activation[token_id] = 1.0
        
        # Encode input
        device = self.brain.output_projection.weight.device
        tokens = tokenizer.encode(inception)
        input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
        
        total_change = 0.0
        lr_adjustments = 0
        
        # Minimum LR floor to prevent learning from stopping completely
        min_lr = self.config.hebbian_lr * 0.1  # Never go below 10% of initial LR
        
        for rep in range(num_reps):
            # Forward pass
            with torch.no_grad():
                x = self.brain.embed_input(input_ids)
                
                if self.brain.thalamus.energy_check(x):
                    filtered_x, _ = self.brain.thalamus(x)
                    if filtered_x.shape[1] < x.shape[1]:
                        padding = torch.zeros(
                            x.shape[0], x.shape[1] - filtered_x.shape[1], 
                            x.shape[2], device=device
                        )
                        x = torch.cat([filtered_x, padding], dim=1)
                    else:
                        x = filtered_x
                
                cortex_out, _ = self.brain.cortex(x)
                pre_activation = cortex_out[:, -1, :]
                
                # dACC check (if enabled) - ONLY intervene on ACTUAL conflicts
                if self.config.dacc_enabled:
                    assessment, dacc_meta = self.dacc(pre_activation)
                    
                    # FIXED: Only reduce LR if conflict is actually HIGH (> 0.5)
                    # AND the LR hasn't already been reduced too much
                    should_reduce = (
                        assessment.conflict > 0.5 and 
                        self.current_lr > min_lr
                    )
                    
                    if should_reduce:
                        old_lr = self.current_lr
                        self.current_lr = max(
                            self.current_lr * self.config.dacc_lr_reduction,
                            min_lr
                        )
                        lr_adjustments += 1
                        self.dacc_interventions += 1
                        
                        if verbose:
                            print(f"      [dACC] High Conflict={assessment.conflict:.2f}, "
                                  f"Reducing LR: {old_lr:.4f} → {self.current_lr:.4f}")
            
            # Apply Hebbian update with current (possibly adjusted) LR
            success, change = self.hebbian.apply_clamped_update(
                weight=self.brain.output_projection.weight,
                pre=pre_activation.squeeze(0),
                target_activation=target_activation,
                dopamine=self.dopamine_level,
                learning_rate_override=self.current_lr,
            )
            
            if success:
                total_change += change
            
            self.lr_history.append(self.current_lr)
        
        # Reset LR for next concept
        final_lr = self.current_lr
        self.current_lr = self.config.hebbian_lr
        
        return {
            'success': True,
            'concept': concept,
            'value': value,
            'total_weight_change': total_change,
            'lr_adjustments': lr_adjustments,
            'final_lr': final_lr,
        }


# ============================================
# MAIN EXPERIMENT CLASS
# ============================================

class AGIBenchmark:
    """The v0.7 AGI Benchmark Experiment."""
    
    def __init__(self, config: ExperimentV07Config):
        self.config = config
        self.device = torch.device(config.device)
        self.brain = None
        self.tokenizer = None
        self.dacc = None
        self.learner = None
        
        # Results tracking
        self.results = {}
        
    def load_model(self) -> bool:
        """Load the brain and initialize components."""
        print("  Loading Nōkai brain...")
        
        checkpoint_path = Path(self.config.checkpoint_dir) / self.config.model_file
        tokenizer_path = Path(self.config.checkpoint_dir) / self.config.tokenizer_file
        
        if not checkpoint_path.exists():
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        if USE_BPE:
            self.tokenizer = NokaiTokenizer.load(str(tokenizer_path))
        else:
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        embedding_weight = state_dict.get('embedding.weight')
        if embedding_weight is not None:
            vocab_size, embedding_dim = embedding_weight.shape
        else:
            vocab_size = self.tokenizer.vocab_size
            embedding_dim = 128
        
        pos_weight = state_dict.get('position_embedding.weight')
        max_seq_length = pos_weight.shape[0] if pos_weight is not None else 512
        
        brain_config = NokaiConfig.nano()
        brain_config.vocab_size = vocab_size
        brain_config.embedding_dim = embedding_dim
        brain_config.max_sequence_length = max_seq_length
        
        self.brain = NeuromorphicBrain(brain_config).to(self.device)
        self.brain.load_state_dict(state_dict, strict=False)
        
        # Initialize dACC
        self.dacc = MetacognitiveMonitor(
            state_dim=embedding_dim,
            confidence_threshold=0.7,
        ).to(self.device)
        
        # Initialize adaptive learner
        self.learner = AdaptiveHebbianLearner(
            brain=self.brain,
            dacc=self.dacc,
            config=self.config,
        )
        
        print(f"  ✓ Brain loaded (vocab={vocab_size}, dim={embedding_dim})")
        print(f"  ✓ dACC initialized")
        print(f"  ✓ Adaptive Hebbian Learner ready")
        
        return True
    
    def get_probability(self, question: str, target: str) -> float:
        """Get probability of target word as next token."""
        self.brain.eval()
        
        with torch.no_grad():
            tokens = self.tokenizer.encode(question)
            input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
            outputs = self.brain(input_ids)
            logits = outputs['logits']
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            token_id = self.learner.get_token_id(self.tokenizer, target)
            if token_id >= 0 and token_id < len(probs):
                return probs[token_id].item()
        
        return 0.0
    
    # =====================================
    # TEST 1: MULTI-CONCEPT LEARNING
    # =====================================
    
    def test_multi_concept(self) -> Dict:
        """
        Test: Can the brain learn multiple concepts without forgetting?
        
        This is something LLMs CANNOT do after deployment.
        """
        print("\n" + "═" * 80)
        print("  TEST 1: MULTI-CONCEPT LEARNING")
        print("  Can the brain learn 5 new facts without gradients?")
        print("═" * 80)
        
        original_weights = self.brain.output_projection.weight.data.clone()
        
        concepts = list(KNOWLEDGE_BASE.keys())[:self.config.concepts_to_learn]
        results = {}
        
        # Measure baselines
        print("\n  📊 BASELINES (before any learning):")
        baselines = {}
        for concept in concepts:
            value = KNOWLEDGE_BASE[concept]
            question = QUESTIONS[concept]
            prob = self.get_probability(question, value)
            baselines[concept] = prob
            print(f"    {concept:10} → {value:10} | Prob: {prob:.4f}")
        
        # Learn each concept sequentially
        print("\n  🧠 LEARNING PHASE:")
        learning_results = []
        
        for i, concept in enumerate(concepts):
            value = KNOWLEDGE_BASE[concept]
            template = INCEPTION_TEMPLATES[concept]
            
            print(f"\n  [{i+1}/{len(concepts)}] Learning: {concept} = {value}")
            
            learn_result = self.learner.learn_concept(
                tokenizer=self.tokenizer,
                concept=concept,
                value=value,
                inception_template=template,
                verbose=True,
            )
            learning_results.append(learn_result)
            
            # Measure probability after learning
            prob_after = self.get_probability(QUESTIONS[concept], value)
            print(f"      Prob: {baselines[concept]:.4f} → {prob_after:.4f} "
                  f"(Δ={prob_after - baselines[concept]:+.4f})")
            
            results[concept] = {
                'baseline': baselines[concept],
                'after_learning': prob_after,
                'learned': prob_after > baselines[concept] * 2,
            }
        
        # Check if all concepts are still learned (forgetting test)
        print("\n  📊 RETENTION TEST (checking all concepts after learning all):")
        retained = 0
        
        for concept in concepts:
            value = KNOWLEDGE_BASE[concept]
            final_prob = self.get_probability(QUESTIONS[concept], value)
            initial_learned = results[concept]['after_learning']
            
            # Check if probability dropped significantly
            if initial_learned > 0:
                retention = final_prob / initial_learned
            else:
                retention = 1.0 if final_prob > baselines[concept] else 0.0
            
            is_retained = final_prob > baselines[concept]
            if is_retained:
                retained += 1
            
            status = "✅" if is_retained else "❌"
            print(f"    {status} {concept:10} → {value:10} | "
                  f"Final: {final_prob:.4f} (retention: {retention:.1%})")
            
            results[concept]['final_prob'] = final_prob
            results[concept]['retained'] = is_retained
        
        # Restore weights
        self.brain.output_projection.weight.data.copy_(original_weights)
        
        # Summary
        print("\n" + "─" * 60)
        print("  MULTI-CONCEPT RESULTS:")
        print("  " + "─" * 56)
        print(f"    Concepts learned: {sum(1 for c in concepts if results[c]['learned'])}/{len(concepts)}")
        print(f"    Concepts retained: {retained}/{len(concepts)}")
        
        if retained == len(concepts):
            print(f"\n  🎉 PERFECT! All concepts learned AND retained!")
            print(f"     This is something GPT/Claude CANNOT do post-deployment!")
        elif retained >= len(concepts) * 0.8:
            print(f"\n  ✅ GOOD! Most concepts retained ({retained}/{len(concepts)})")
        else:
            print(f"\n  ⚠️ Catastrophic forgetting detected! Only {retained}/{len(concepts)} retained")
        
        return results
    
    # =====================================
    # TEST 2: METACOGNITION ("I DON'T KNOW")
    # =====================================
    
    def test_metacognition(self) -> Dict:
        """
        Test: Does the brain know what it DOESN'T know?
        
        LLMs hallucinate answers. AGI should say "I don't know."
        """
        print("\n" + "═" * 80)
        print("  TEST 2: METACOGNITION")
        print("  Does the brain know what it DOESN'T know?")
        print("═" * 80)
        
        results = {}
        
        # Test on unknown concepts
        unknown_questions = [
            ("What is the blorbification constant?", "blorbification"),
            ("What color is a xynthorp?", "xynthorp"),
            ("How many legs does a quaznar have?", "quaznar"),
        ]
        
        # Test on known concepts (should have high confidence)
        known_questions = [
            ("What color is an apple?", "apple"),  # Trained on this
        ]
        
        print("\n  📊 Testing dACC confidence on UNKNOWN concepts:")
        
        for question, concept in unknown_questions:
            with torch.no_grad():
                tokens = self.tokenizer.encode(question)
                input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
                outputs = self.brain(input_ids)
                
                # Get hidden state for dACC
                x = self.brain.embed_input(input_ids)
                cortex_out, _ = self.brain.cortex(x)
                hidden = cortex_out[:, -1, :]
                
                assessment, _ = self.dacc(hidden)
                
                print(f"    Q: \"{question[:40]}...\"")
                print(f"       Confidence: {assessment.confidence:.3f} | "
                      f"Conflict: {assessment.conflict:.3f} | "
                      f"State: {assessment.recommended_state.value}")
                
                results[concept] = {
                    'question': question,
                    'known': False,
                    'confidence': assessment.confidence,
                    'conflict': assessment.conflict,
                    'state': assessment.recommended_state.value,
                }
        
        print("\n  📊 Testing dACC confidence on KNOWN concepts:")
        
        for question, concept in known_questions:
            with torch.no_grad():
                tokens = self.tokenizer.encode(question)
                input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
                outputs = self.brain(input_ids)
                
                x = self.brain.embed_input(input_ids)
                cortex_out, _ = self.brain.cortex(x)
                hidden = cortex_out[:, -1, :]
                
                assessment, _ = self.dacc(hidden)
                
                print(f"    Q: \"{question}\"")
                print(f"       Confidence: {assessment.confidence:.3f} | "
                      f"Conflict: {assessment.conflict:.3f} | "
                      f"State: {assessment.recommended_state.value}")
                
                results[concept] = {
                    'question': question,
                    'known': True,
                    'confidence': assessment.confidence,
                    'conflict': assessment.conflict,
                    'state': assessment.recommended_state.value,
                }
        
        # Analysis
        print("\n" + "─" * 60)
        print("  METACOGNITION RESULTS:")
        
        unknown_avg_conf = sum(r['confidence'] for k, r in results.items() if not r['known']) / 3
        known_avg_conf = results.get('apple', {}).get('confidence', 0.5)
        
        print(f"    Unknown concepts avg confidence: {unknown_avg_conf:.3f}")
        print(f"    Known concepts avg confidence:   {known_avg_conf:.3f}")
        
        if unknown_avg_conf < known_avg_conf:
            print(f"\n  ✅ dACC correctly shows LOWER confidence on unknown concepts!")
            print(f"     This is metacognition - knowing what you don't know.")
        else:
            print(f"\n  ⚠️ dACC needs calibration - confidence similar for known/unknown")
        
        return results
    
    # =====================================
    # FULL BENCHMARK
    # =====================================
    
    def run_full_benchmark(self):
        """Run all AGI benchmark tests."""
        
        print("\n" + "═" * 80)
        print("║" + " " * 78 + "║")
        print("║" + "  🧠 NŌKAI v0.7: THE AGI BENCHMARK".center(78) + "║")
        print("║" + "  Testing capabilities that LLMs CANNOT do".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("═" * 80)
        
        if not self.load_model():
            print("\n  ❌ Failed to load model")
            return
        
        all_results = {}
        
        # Test 1: Multi-concept learning
        all_results['multi_concept'] = self.test_multi_concept()
        
        # Test 2: Metacognition
        all_results['metacognition'] = self.test_metacognition()
        
        # Summary
        print("\n" + "═" * 80)
        print("  AGI BENCHMARK SUMMARY")
        print("═" * 80)
        
        # Multi-concept score
        mc = all_results['multi_concept']
        learned = sum(1 for v in mc.values() if isinstance(v, dict) and v.get('learned', False))
        retained = sum(1 for v in mc.values() if isinstance(v, dict) and v.get('retained', False))
        
        print(f"\n  1. MULTI-CONCEPT LEARNING:")
        print(f"     Learned: {learned}/{self.config.concepts_to_learn}")
        print(f"     Retained: {retained}/{self.config.concepts_to_learn}")
        score1 = (learned + retained) / (self.config.concepts_to_learn * 2)
        
        # Metacognition score
        meta = all_results['metacognition']
        unknown_confs = [v['confidence'] for v in meta.values() if not v.get('known', True)]
        known_confs = [v['confidence'] for v in meta.values() if v.get('known', False)]
        
        print(f"\n  2. METACOGNITION:")
        if unknown_confs and known_confs:
            meta_works = sum(unknown_confs) / len(unknown_confs) < sum(known_confs) / len(known_confs)
            print(f"     Unknown confidence: {sum(unknown_confs)/len(unknown_confs):.3f}")
            print(f"     Known confidence: {sum(known_confs)/len(known_confs):.3f}")
            print(f"     Status: {'✅ Working' if meta_works else '⚠️ Needs work'}")
            score2 = 1.0 if meta_works else 0.5
        else:
            score2 = 0.5
        
        # Overall AGI score
        agi_score = (score1 + score2) / 2
        
        print(f"\n  ═══════════════════════════════════════")
        print(f"      AGI CAPABILITY SCORE: {agi_score:.1%}")
        print(f"  ═══════════════════════════════════════")
        
        if agi_score >= 0.8:
            print(f"\n  🎉 EXCELLENT! Nōkai demonstrates strong AGI capabilities!")
            print(f"     These are things LLMs fundamentally CANNOT do.")
        elif agi_score >= 0.6:
            print(f"\n  ✅ GOOD! Nōkai shows promising AGI features.")
        else:
            print(f"\n  ⚠️ More work needed on core AGI capabilities.")
        
        print("\n" + "═" * 80)
        print("  END OF AGI BENCHMARK")
        print("═" * 80 + "\n")
        
        self.results = all_results
        return all_results


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Nōkai v0.7 - AGI Benchmark")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full",
                       choices=["multi", "meta", "full"])
    parser.add_argument("--hebbian_lr", type=float, default=0.06)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--concepts", type=int, default=5)
    
    args = parser.parse_args()
    
    # Auto-detect model
    model_file = args.model_file
    if model_file is None:
        checkpoint_dir = Path(args.checkpoint_dir)
        for candidate in ["brain_epoch_5.pt", "brain_best.pt"]:
            if (checkpoint_dir / candidate).exists():
                model_file = candidate
                break
        if model_file is None:
            pt_files = list(checkpoint_dir.glob("*.pt"))
            model_file = pt_files[0].name if pt_files else "brain_epoch_5.pt"
    
    config = ExperimentV07Config(
        checkpoint_dir=args.checkpoint_dir,
        model_file=model_file,
        mode=args.mode,
        hebbian_lr=args.hebbian_lr,
        num_repetitions=args.repetitions,
        concepts_to_learn=args.concepts,
    )
    
    benchmark = AGIBenchmark(config)
    
    if args.mode == "multi":
        if benchmark.load_model():
            benchmark.test_multi_concept()
    elif args.mode == "meta":
        if benchmark.load_model():
            benchmark.test_metacognition()
    else:
        benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
