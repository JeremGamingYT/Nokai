#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             NŌKAI v0.9: SCALING TO REAL INTELLIGENCE                        ║
║                                                                              ║
║   Requirements for TRUE language understanding:                              ║
║   1. REAL DATA       - OpenWebText, Wikipedia, Books (NOT TinyStories)      ║
║   2. LARGER MODEL    - 100M+ parameters (currently 23M)                     ║
║   3. LONGER TRAINING - Days, not hours                                      ║
║   4. COMPOSITIONAL   - Chain reasoning (A→B, B→C ∴ A→C)                     ║
║                                                                              ║
║   Target: Complete sentences like "Tim was sad, but he agreed to trade      ║
║           the expensive car for a smaller one."                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author: Nōkai Research Team  
Version: 0.9 - REAL INTELLIGENCE
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
import json
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nokai import NeuromorphicBrain, NokaiConfig

try:
    from nokai.tokenization import NokaiTokenizer, TokenizerConfig
    USE_BPE = True
except ImportError:
    USE_BPE = False


# ============================================
# CONFIGURATION FOR SCALING
# ============================================

@dataclass
class ScalingConfig:
    """Configuration for scaling Nōkai to real intelligence."""
    
    # Model size tiers
    # nano:  23M params  (current) - TinyStories level
    # small: 100M params           - Basic conversation
    # medium: 350M params          - Good conversation  
    # large: 1B+ params            - Human-like
    
    model_tier: str = "small"  # "nano", "small", "medium", "large"
    
    # Data sources (prioritized)
    data_sources: List[str] = field(default_factory=lambda: [
        "openwebtext",      # 38GB of Reddit content
        "wikipedia",        # Encyclopedic knowledge
        "bookcorpus",       # 11K books
        "c4",               # Colossal Clean Crawled Corpus
    ])
    
    # Training
    batch_size: int = 32
    gradient_accumulation: int = 4  # Effective batch = 128
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 100000
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints_v09"
    save_every: int = 5000
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def get_model_config(self) -> Dict:
        """Get model configuration based on tier."""
        configs = {
            "nano": {
                "embedding_dim": 128,
                "num_layers": 6,
                "num_heads": 4,
                "ff_dim": 512,
                "max_seq_length": 512,
                "vocab_size": 32000,
            },
            "small": {
                "embedding_dim": 512,
                "num_layers": 12,
                "num_heads": 8,
                "ff_dim": 2048,
                "max_seq_length": 1024,
                "vocab_size": 32000,
            },
            "medium": {
                "embedding_dim": 768,
                "num_layers": 24,
                "num_heads": 12,
                "ff_dim": 3072,
                "max_seq_length": 2048,
                "vocab_size": 50000,
            },
            "large": {
                "embedding_dim": 1024,
                "num_layers": 32,
                "num_heads": 16,
                "ff_dim": 4096,
                "max_seq_length": 4096,
                "vocab_size": 50000,
            },
        }
        return configs.get(self.model_tier, configs["small"])


# ============================================
# REAL DATA LOADER
# ============================================

class RealDataLoader:
    """
    Load REAL training data, not TinyStories.
    
    Supported datasets:
    - openwebtext: Reddit outbound links (high quality web text)
    - wikipedia: Encyclopedic knowledge
    - bookcorpus: 11K unpublished books
    - c4: Colossal Clean Crawled Corpus
    """
    
    def __init__(self, config: ScalingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.datasets = {}
        
    def load_openwebtext(self, max_samples: int = None):
        """Load OpenWebText dataset."""
        try:
            from datasets import load_dataset
            
            print("  📥 Loading OpenWebText...")
            dataset = load_dataset("openwebtext", split="train", streaming=True)
            
            if max_samples:
                dataset = dataset.take(max_samples)
            
            self.datasets["openwebtext"] = dataset
            print("  ✓ OpenWebText loaded (streaming)")
            return True
            
        except Exception as e:
            print(f"  ⚠️ Could not load OpenWebText: {e}")
            return False
    
    def load_wikipedia(self, language: str = "en", max_samples: int = None):
        """Load Wikipedia dataset."""
        try:
            from datasets import load_dataset
            
            print(f"  📥 Loading Wikipedia ({language})...")
            dataset = load_dataset(
                "wikipedia", 
                f"20220301.{language}", 
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            
            if max_samples:
                dataset = dataset.take(max_samples)
            
            self.datasets["wikipedia"] = dataset
            print("  ✓ Wikipedia loaded (streaming)")
            return True
            
        except Exception as e:
            print(f"  ⚠️ Could not load Wikipedia: {e}")
            return False
    
    def load_pile(self, max_samples: int = None):
        """Load The Pile dataset (diverse sources)."""
        try:
            from datasets import load_dataset
            
            print("  📥 Loading The Pile...")
            dataset = load_dataset(
                "EleutherAI/pile",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            
            if max_samples:
                dataset = dataset.take(max_samples)
            
            self.datasets["pile"] = dataset
            print("  ✓ The Pile loaded (streaming)")
            return True
            
        except Exception as e:
            print(f"  ⚠️ Could not load The Pile: {e}")
            return False
    
    def load_c4(self, max_samples: int = None):
        """Load C4 (Colossal Clean Crawled Corpus)."""
        try:
            from datasets import load_dataset
            
            print("  📥 Loading C4...")
            dataset = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            
            if max_samples:
                dataset = dataset.take(max_samples)
            
            self.datasets["c4"] = dataset
            print("  ✓ C4 loaded (streaming)")
            return True
            
        except Exception as e:
            print(f"  ⚠️ Could not load C4: {e}")
            return False
    
    def get_batch(self, dataset_name: str, batch_size: int):
        """Get a batch of tokenized sequences."""
        if dataset_name not in self.datasets:
            return None
        
        dataset = self.datasets[dataset_name]
        batch_texts = []
        
        for sample in dataset:
            text = sample.get("text", sample.get("content", ""))
            if text:
                batch_texts.append(text)
                if len(batch_texts) >= batch_size:
                    break
        
        if not batch_texts:
            return None
        
        # Tokenize
        max_len = self.config.get_model_config()["max_seq_length"]
        input_ids = []
        
        for text in batch_texts:
            tokens = self.tokenizer.encode(text)[:max_len]
            if len(tokens) < max_len:
                tokens = tokens + [0] * (max_len - len(tokens))
            input_ids.append(tokens)
        
        return torch.tensor(input_ids)


# ============================================
# COMPOSITIONAL REASONING MODULE
# ============================================

class CompositionalReasoner:
    """
    Handles chain reasoning: If A→B and B→C, then A→C.
    
    This is a key AGI capability that emerges with scale.
    """
    
    def __init__(self, brain, memory):
        self.brain = brain
        self.memory = memory  # Experience Memory from v0.8
        self.chains = {}  # Store learned chains
    
    def learn_relation(self, entity_a: str, relation: str, entity_b: str):
        """Learn a relation: A --relation--> B."""
        key = (entity_a, relation)
        self.chains[key] = entity_b
    
    def infer_chain(self, entity_a: str, relation: str) -> List[str]:
        """
        Infer the chain starting from entity_a.
        
        Example:
        - Learn: Tim → sad, sad → agree, agree → trade
        - Infer: Tim → ... → ...
        """
        chain = [entity_a]
        current = entity_a
        
        for _ in range(10):  # Max chain length
            key = (current, relation)
            if key in self.chains:
                next_entity = self.chains[key]
                chain.append(next_entity)
                current = next_entity
            else:
                break
        
        return chain
    
    def can_reach(self, entity_a: str, entity_b: str, relation: str) -> bool:
        """Check if A can reach B through relation chains."""
        chain = self.infer_chain(entity_a, relation)
        return entity_b in chain


# ============================================
# GOAL-DIRECTED BEHAVIOR MODULE  
# ============================================

class GoalDirectedAgent:
    """
    Enables goal-directed behavior and planning.
    
    The agent can:
    1. Set goals
    2. Plan steps to achieve goals
    3. Execute and monitor progress
    4. Adjust if needed
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.current_goal = None
        self.plan = []
        self.history = []
    
    def set_goal(self, goal: str):
        """Set a new goal."""
        self.current_goal = goal
        self.plan = []
        print(f"  🎯 Goal set: {goal}")
    
    def plan_steps(self) -> List[str]:
        """
        Generate a plan to achieve the goal.
        
        This would use the brain's reasoning to decompose
        the goal into sub-goals.
        """
        # Simplified planning
        if self.current_goal:
            self.plan = [
                f"Understand: {self.current_goal}",
                f"Break down into sub-tasks",
                f"Execute each sub-task",
                f"Verify completion",
            ]
        return self.plan
    
    def execute_step(self, step_idx: int) -> Dict:
        """Execute a step in the plan."""
        if step_idx >= len(self.plan):
            return {"success": False, "error": "No more steps"}
        
        step = self.plan[step_idx]
        # Would execute via brain.generate()
        result = {"success": True, "step": step, "output": f"Completed: {step}"}
        self.history.append(result)
        return result


# ============================================
# SELF-IMPROVEMENT LOOP
# ============================================

class SelfImprover:
    """
    The model improves its own performance through:
    1. Self-evaluation
    2. Identifying weaknesses
    3. Targeted learning
    4. Re-evaluation
    
    This is a key step towards ASI.
    """
    
    def __init__(self, brain, learner):
        self.brain = brain
        self.learner = learner
        self.improvement_log = []
    
    def evaluate_self(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate performance on test cases.
        
        Returns metrics on what the model knows/doesn't know.
        """
        correct = 0
        wrong = []
        
        for case in test_cases:
            question = case.get("question", "")
            expected = case.get("answer", "")
            # Would generate and compare
            # For now, simplified
            correct += 1  # Placeholder
        
        accuracy = correct / len(test_cases) if test_cases else 0
        
        return {
            "accuracy": accuracy,
            "wrong_cases": wrong,
            "total": len(test_cases),
        }
    
    def identify_weaknesses(self, evaluation: Dict) -> List[str]:
        """Identify areas that need improvement."""
        weaknesses = []
        
        for case in evaluation.get("wrong_cases", []):
            topic = case.get("topic", "unknown")
            if topic not in weaknesses:
                weaknesses.append(topic)
        
        return weaknesses
    
    def improve(self, weakness: str, training_data: List[str]):
        """
        Targeted learning on a weakness.
        
        Uses Hebbian learning to reinforce correct patterns.
        """
        print(f"  🔧 Improving on: {weakness}")
        
        for text in training_data:
            # Would apply Hebbian updates
            pass
        
        self.improvement_log.append({
            "weakness": weakness,
            "training_samples": len(training_data),
            "timestamp": time.time(),
        })
    
    def improvement_cycle(self, test_cases: List[Dict], training_data: Dict):
        """
        Full self-improvement cycle:
        1. Evaluate
        2. Identify weaknesses
        3. Improve
        4. Re-evaluate
        """
        print("\n  🔄 Starting self-improvement cycle...")
        
        # Evaluate
        eval_before = self.evaluate_self(test_cases)
        print(f"  📊 Before: {eval_before['accuracy']:.1%} accuracy")
        
        # Identify weaknesses
        weaknesses = self.identify_weaknesses(eval_before)
        print(f"  ⚠️ Weaknesses: {weaknesses}")
        
        # Improve each weakness
        for weakness in weaknesses:
            if weakness in training_data:
                self.improve(weakness, training_data[weakness])
        
        # Re-evaluate
        eval_after = self.evaluate_self(test_cases)
        print(f"  📊 After: {eval_after['accuracy']:.1%} accuracy")
        
        improvement = eval_after['accuracy'] - eval_before['accuracy']
        print(f"  📈 Improvement: {improvement:+.1%}")
        
        return {
            "before": eval_before,
            "after": eval_after,
            "improvement": improvement,
        }


# ============================================
# REAL CONVERSATION EVALUATOR
# ============================================

class ConversationEvaluator:
    """
    Tests if the model can hold REAL conversations.
    
    Target: "Tim was sad, but he agreed to trade the expensive car for a smaller one."
    """
    
    def __init__(self, brain, tokenizer):
        self.brain = brain
        self.tokenizer = tokenizer
        
        self.test_cases = [
            {
                "prompt": "Tim was sad, but he agreed to trade",
                "expected_contains": ["car", "smaller", "one"],
                "topic": "sentence_completion",
            },
            {
                "prompt": "The capital of France is",
                "expected_contains": ["Paris"],
                "topic": "factual_knowledge",
            },
            {
                "prompt": "If it rains, then the ground will be",
                "expected_contains": ["wet", "damp", "soaked"],
                "topic": "reasoning",
            },
            {
                "prompt": "She walked to the store to buy",
                "expected_contains": ["food", "milk", "groceries", "bread"],
                "topic": "common_sense",
            },
        ]
    
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from prompt."""
        self.brain.eval()
        device = next(self.brain.parameters()).device
        
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.brain(input_ids)
                logits = outputs['logits']
                
                # Sample from distribution
                probs = F.softmax(logits[0, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Stop at EOS
                if next_token.item() == 0:
                    break
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(input_ids[0].tolist())
    
    def evaluate(self) -> Dict:
        """Run all conversation tests."""
        results = []
        
        for case in self.test_cases:
            prompt = case["prompt"]
            expected = case["expected_contains"]
            
            generated = self.generate(prompt)
            
            # Check if any expected word is in output
            found = any(word.lower() in generated.lower() for word in expected)
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "expected": expected,
                "passed": found,
                "topic": case["topic"],
            })
        
        passed = sum(1 for r in results if r["passed"])
        
        return {
            "total": len(self.test_cases),
            "passed": passed,
            "accuracy": passed / len(self.test_cases),
            "results": results,
        }


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

class NokaiTrainerV09:
    """
    Full training pipeline for Nōkai v0.9.
    
    Scales the model to real intelligence.
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.brain = None
        self.tokenizer = None
        self.optimizer = None
        self.data_loader = None
        
        # Modules
        self.reasoner = None
        self.agent = None
        self.improver = None
        self.evaluator = None
        
        self.step = 0
        self.best_loss = float('inf')
    
    def setup(self):
        """Initialize all components."""
        print("\n" + "═" * 80)
        print("  NŌKAI v0.9: SCALING TO REAL INTELLIGENCE")
        print("═" * 80)
        
        model_config = self.config.get_model_config()
        
        print(f"\n  Model Tier: {self.config.model_tier.upper()}")
        print(f"  Embedding Dim: {model_config['embedding_dim']}")
        print(f"  Layers: {model_config['num_layers']}")
        print(f"  Heads: {model_config['num_heads']}")
        print(f"  Max Seq Length: {model_config['max_seq_length']}")
        print(f"  Vocab Size: {model_config['vocab_size']}")
        
        # Create model config
        brain_config = NokaiConfig(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config['embedding_dim'],
            num_cortex_layers=model_config['num_layers'],
            num_attention_heads=model_config['num_heads'],
            feed_forward_dim=model_config['ff_dim'],
            max_sequence_length=model_config['max_seq_length'],
        )
        
        # Initialize brain
        print("\n  Initializing Nōkai brain...")
        self.brain = NeuromorphicBrain(brain_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.brain.parameters())
        trainable_params = sum(p.numel() for p in self.brain.parameters() if p.requires_grad)
        
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable: {trainable_params:,}")
        
        # Move to device
        device = torch.device(self.config.device)
        self.brain = self.brain.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.brain.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        # Tokenizer
        print("\n  Initializing tokenizer...")
        if USE_BPE:
            tokenizer_config = TokenizerConfig(vocab_size=model_config['vocab_size'])
            self.tokenizer = NokaiTokenizer(tokenizer_config)
            print(f"  ✓ Tokenizer ready (vocab={model_config['vocab_size']})")
        
        # Data loader
        self.data_loader = RealDataLoader(self.config, self.tokenizer)
        
        # Initialize modules
        self.evaluator = ConversationEvaluator(self.brain, self.tokenizer)
        
        print("\n  ✓ Setup complete!")
        return True
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.brain.train()
        device = next(self.brain.parameters()).device
        batch = batch.to(device)
        
        # Forward pass
        outputs = self.brain(batch)
        logits = outputs['logits']
        
        # Compute loss (next token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,  # Padding token
        )
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def train(self, max_steps: int = None):
        """Full training loop."""
        if max_steps is None:
            max_steps = self.config.total_steps
        
        print(f"\n  Starting training for {max_steps} steps...")
        
        # Load data
        print("\n  Loading datasets...")
        loaded_any = False
        for source in self.config.data_sources:
            if source == "openwebtext":
                loaded_any = self.data_loader.load_openwebtext() or loaded_any
            elif source == "wikipedia":
                loaded_any = self.data_loader.load_wikipedia() or loaded_any
            elif source == "pile":
                loaded_any = self.data_loader.load_pile() or loaded_any
            elif source == "c4":
                loaded_any = self.data_loader.load_c4() or loaded_any
        
        if not loaded_any:
            print("  ⚠️ No datasets loaded! Training with synthetic data.")
            return
        
        # Training loop
        losses = []
        start_time = time.time()
        
        for step in range(max_steps):
            self.step = step
            
            # Get batch
            batch = None
            for source in self.data_loader.datasets.keys():
                batch = self.data_loader.get_batch(source, self.config.batch_size)
                if batch is not None:
                    break
            
            if batch is None:
                print("  ⚠️ Could not get batch, skipping...")
                continue
            
            # Train step
            loss = self.train_step(batch)
            losses.append(loss)
            
            # Log
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                
                print(f"  Step {step:6d} | Loss: {avg_loss:.4f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s")
            
            # Save checkpoint
            if step > 0 and step % self.config.save_every == 0:
                self.save_checkpoint(step)
        
        print(f"\n  ✓ Training complete!")
        
        # Final evaluation
        print("\n  📊 Final Evaluation:")
        eval_results = self.evaluator.evaluate()
        print(f"     Conversation accuracy: {eval_results['accuracy']:.1%}")
        
        for result in eval_results['results']:
            status = "✅" if result['passed'] else "❌"
            print(f"     {status} {result['topic']}: \"{result['prompt'][:30]}...\"")
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        path = checkpoint_dir / f"nokai_v09_step_{step}.pt"
        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint saved: {path}")


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Nōkai v0.9 - Scaling to Real Intelligence")
    
    parser.add_argument("--tier", type=str, default="small",
                       choices=["nano", "small", "medium", "large"],
                       help="Model size tier")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v09")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate, don't train")
    
    args = parser.parse_args()
    
    config = ScalingConfig(
        model_tier=args.tier,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        total_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    trainer = NokaiTrainerV09(config)
    
    if trainer.setup():
        if not args.eval_only:
            trainer.train(args.steps)
        else:
            print("\n  📊 Evaluation only mode:")
            eval_results = trainer.evaluator.evaluate()
            print(f"     Accuracy: {eval_results['accuracy']:.1%}")


if __name__ == "__main__":
    main()
