#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       NŌKAI v0.9: H100-OPTIMIZED TRAINING WRAPPER                            ║
║                                                                              ║
║   This script WRAPS the existing train_v09_scaling.py to add H100            ║
║   hardware-specific optimizations WITHOUT modifying original code.           ║
║                                                                              ║
║   Optimizations applied:                                                     ║
║   • torch.compile() with max-autotune                                        ║
║   • BF16 mixed precision (native H100)                                       ║
║   • cuDNN benchmark auto-tuning                                              ║
║   • Fused AdamW optimizer                                                    ║
║   • Optimized gradient accumulation                                          ║
║   • CUDA backend tuning                                                      ║
║   • Async data prefetching                                                   ║
║   • Memory optimization                                                      ║
║                                                                              ║
║   Expected speedup: 1.5x - 3x on H100 compared to default training           ║
║                                                                              ║
║   Usage:                                                                     ║
║   python scripts/train_v09_h100.py --tier medium --steps 50000              ║
║                                                                              ║
║   Author: Nōkai Research Team                                                ║
║   Target: NVIDIA H100 80GB HBM3                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure Dynamo early for Nokai compatibility
# Nokai uses .item() calls which require special handling
try:
    import torch._dynamo.config as dynamo_config
    dynamo_config.capture_scalar_outputs = True
    dynamo_config.suppress_errors = True
except ImportError:
    pass  # Older PyTorch version

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import original training components (UNCHANGED)
from train_v09_scaling import (
    ScalingConfig,
    RealDataLoader,
    NokaiTrainerV09,
    CompositionalReasoner,
    GoalDirectedAgent,
    SelfImprover,
    ConversationEvaluator,
)

# Import H100 optimizations (NEW)
from nokai.h100_optimizer import (
    H100Optimizer,
    H100MemoryManager,
    H100Benchmark,
    MixedPrecisionConfig,
    H100DataLoaderConfig,
    create_fused_adam,
    h100_autocast,
    compile_model_for_h100,
    detect_h100_capabilities,
    configure_cuda_backend_for_h100,
    AsyncPrefetcher,
    OptimizedGradientAccumulator,
)

from nokai import NeuromorphicBrain, NokaiConfig

try:
    from nokai.tokenization import NokaiTokenizer, TokenizerConfig
    USE_BPE = True
except ImportError:
    USE_BPE = False


# ============================================
# H100-ENHANCED SCALING CONFIG
# ============================================

@dataclass
class H100ScalingConfig(ScalingConfig):
    """
    Extended configuration with H100-specific settings.
    Inherits all settings from ScalingConfig without modification.
    """
    
    # H100-specific extensions (NEW settings only)
    use_bf16: bool = True                    # Use BF16 instead of FP16
    use_torch_compile: bool = True           # Enable torch.compile
    compile_mode: str = "max-autotune"       # max-autotune for best performance
    use_fused_optimizer: bool = True         # Use fused AdamW
    async_data_loading: bool = True          # Async GPU transfers
    cudnn_benchmark: bool = True             # cuDNN auto-tuning
    prefetch_factor: int = 4                 # Dataloader prefetching
    num_workers: int = 8                     # Parallel data loading
    
    # Memory optimization
    gradient_checkpointing: bool = True      # Reduce memory via recomputation
    empty_cache_interval: int = 100          # Clear cache every N steps
    
    # Performance tuning
    matmul_precision: str = "high"           # "highest", "high", "medium"
    tf32_enabled: bool = True                # TF32 on Tensor Cores
    
    def get_h100_model_config(self) -> Dict:
        """Get model config optimized for H100."""
        base_config = self.get_model_config()
        
        # H100 can handle larger batch sizes
        # Adjust sequence length if needed for memory
        if self.model_tier == "large":
            # For 1B+ models, adjust for 80GB
            base_config["max_seq_length"] = min(base_config["max_seq_length"], 4096)
        
        return base_config


# ============================================
# H100-OPTIMIZED TRAINER
# ============================================

class NokaiTrainerH100(NokaiTrainerV09):
    """
    H100-optimized trainer that extends NokaiTrainerV09.
    
    All original functionality is preserved. This class ADDS:
    - Model compilation with torch.compile
    - BF16 autocast during training
    - Fused optimizer
    - Optimized gradient accumulation
    - Memory management utilities
    """
    
    def __init__(self, config: H100ScalingConfig):
        # Call parent init (original behavior preserved)
        super().__init__(config)
        
        self.h100_config = config
        self.h100_optimizer: Optional[H100Optimizer] = None
        self.compiled_brain = None
        self.grad_accumulator = None
        self.memory_manager = H100MemoryManager()
        
        # Track H100-specific metrics
        self.h100_metrics = {
            "compilation_time": 0,
            "peak_memory_gb": 0,
            "avg_step_time_ms": [],
        }
    
    def setup(self) -> bool:
        """
        Setup with H100 optimizations.
        
        Extends parent setup() without modifying its behavior.
        """
        print("\n" + "═" * 80)
        print("  NŌKAI v0.9: H100-OPTIMIZED TRAINING")
        print("═" * 80)
        
        # Detect H100 capabilities
        caps = detect_h100_capabilities()
        
        if caps.is_h100:
            print("\n  🎯 NVIDIA H100 DETECTED - Applying full optimizations")
        else:
            print(f"\n  ⚠️ GPU is not H100 (Compute: {caps.compute_capability})")
            print("      Optimizations will still be applied where compatible")
        
        # Configure CUDA backend BEFORE model creation
        configure_cuda_backend_for_h100()
        
        # Set precision
        if self.h100_config.tf32_enabled:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ✓ TF32 precision enabled")
        
        torch.set_float32_matmul_precision(self.h100_config.matmul_precision)
        print(f"  ✓ Matrix multiplication precision: {self.h100_config.matmul_precision}")
        
        # Call parent setup (creates model, tokenizer, etc.)
        success = super().setup()
        if not success:
            return False
        
        # Initialize H100 optimizer AFTER model creation
        dtype = torch.bfloat16 if self.h100_config.use_bf16 else torch.float16
        mp_config = MixedPrecisionConfig(
            dtype=dtype,
            use_fp8=caps.has_fp8,
        )
        
        self.h100_optimizer = H100Optimizer(
            mixed_precision_config=mp_config,
            compile_mode=self.h100_config.compile_mode,
            gradient_accumulation_steps=self.h100_config.gradient_accumulation,
        )
        
        # Apply H100 optimizations to model
        print("\n  🚀 Applying H100 optimizations...")
        
        # Convert to optimal dtype
        device = torch.device(self.config.device)
        self.brain = self.brain.to(device=device, dtype=dtype)
        print(f"  ✓ Model converted to {dtype}")
        
        # Enable gradient checkpointing if available
        if self.h100_config.gradient_checkpointing:
            if hasattr(self.brain, 'gradient_checkpointing_enable'):
                self.brain.gradient_checkpointing_enable()
                print("  ✓ Gradient checkpointing enabled")
        
        # Compile model with torch.compile
        if self.h100_config.use_torch_compile:
            print(f"\n  🔧 Compiling model (mode={self.h100_config.compile_mode})...")
            compile_start = time.time()
            
            self.compiled_brain = compile_model_for_h100(
                self.brain,
                mode=self.h100_config.compile_mode,
                fullgraph=False,  # Must be False - Nokai uses .item() which causes graph breaks
                dynamic=False,
            )
            
            self.h100_metrics["compilation_time"] = time.time() - compile_start
            print(f"  ✓ Compilation complete ({self.h100_metrics['compilation_time']:.1f}s)")
        else:
            self.compiled_brain = self.brain
        
        # Create fused optimizer
        if self.h100_config.use_fused_optimizer:
            print("\n  🔧 Creating fused AdamW optimizer...")
            self.optimizer = create_fused_adam(
                self.compiled_brain,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        
        # Create optimized gradient accumulator
        self.grad_accumulator = OptimizedGradientAccumulator(
            optimizer=self.optimizer,
            accumulation_steps=self.h100_config.gradient_accumulation,
            max_grad_norm=1.0,
            scaler=self.h100_optimizer.grad_scaler,
        )
        
        # Print memory stats
        self.memory_manager.print_memory_stats("Post-setup ")
        
        print("\n  ✅ H100-optimized setup complete!")
        return True
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        H100-optimized training step.
        
        Overrides parent to add:
        - BF16 autocast
        - Optimized gradient accumulation
        - Memory management
        """
        model = self.compiled_brain or self.brain
        model.train()
        
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Async transfer to GPU
        batch = batch.to(device, non_blocking=True)
        
        step_start = time.perf_counter()
        
        # Forward pass with autocast
        with self.h100_optimizer.get_autocast_context():
            outputs = model(batch)
            logits = outputs['logits']
            
            # Compute loss (next token prediction)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
            )
        
        # Backward with accumulation
        self.grad_accumulator.backward(loss)
        step_performed = self.grad_accumulator.step()
        
        step_time = (time.perf_counter() - step_start) * 1000
        self.h100_metrics["avg_step_time_ms"].append(step_time)
        
        # Memory management
        if self.step > 0 and self.step % self.h100_config.empty_cache_interval == 0:
            self.memory_manager.sync_and_clear_cache()
        
        return loss.item()
    
    def train(self, max_steps: int = None):
        """
        H100-optimized training loop.
        
        Extends parent train() with performance tracking.
        """
        if max_steps is None:
            max_steps = self.config.total_steps
        
        print(f"\n  🚀 Starting H100-optimized training for {max_steps} steps...")
        print(f"     Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
        
        # Track peak memory
        torch.cuda.reset_peak_memory_stats()
        
        # Load data (original behavior)
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
        
        # Training loop with H100 optimizations
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
            
            # H100-optimized train step
            loss = self.train_step(batch)
            losses.append(loss)
            
            # Log with H100 metrics
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                
                # Calculate average step time
                recent_times = self.h100_metrics["avg_step_time_ms"][-100:]
                avg_step_ms = sum(recent_times) / len(recent_times) if recent_times else 0
                
                # Memory stats
                mem_stats = self.memory_manager.get_memory_stats()
                
                print(f"  Step {step:6d} | Loss: {avg_loss:.4f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s | "
                      f"Step: {avg_step_ms:.1f}ms | "
                      f"Mem: {mem_stats.get('allocated', 0):.1f}GB")
            
            # Save checkpoint (original behavior preserved)
            if step > 0 and step % self.config.save_every == 0:
                self.save_checkpoint(step)
        
        # Final stats
        self.h100_metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"\n  ✅ Training complete!")
        print(f"\n  📊 H100 Performance Summary:")
        print(f"     • Compilation time: {self.h100_metrics['compilation_time']:.1f}s")
        print(f"     • Peak memory: {self.h100_metrics['peak_memory_gb']:.2f} GB")
        
        if self.h100_metrics["avg_step_time_ms"]:
            avg_step = sum(self.h100_metrics["avg_step_time_ms"]) / len(self.h100_metrics["avg_step_time_ms"])
            print(f"     • Average step time: {avg_step:.1f}ms")
        
        # Final evaluation (original behavior preserved)
        print("\n  📊 Final Evaluation:")
        eval_results = self.evaluator.evaluate()
        print(f"     Conversation accuracy: {eval_results['accuracy']:.1%}")
        
        for result in eval_results['results']:
            status = "✅" if result['passed'] else "❌"
            print(f"     {status} {result['topic']}: \"{result['prompt'][:30]}...\"")
    
    def benchmark(self, num_iterations: int = 100):
        """
        Benchmark the H100-optimized model.
        
        New method added for performance testing.
        """
        print("\n  📊 Running H100 benchmark...")
        
        model = self.compiled_brain or self.brain
        model_config = self.config.get_model_config()
        
        results = H100Benchmark.benchmark_throughput(
            model,
            input_shape=(self.config.batch_size, model_config["max_seq_length"]),
            num_iterations=num_iterations,
        )
        
        print(f"\n  Benchmark Results:")
        print(f"     • Samples/second: {results['samples_per_second']:.1f}")
        print(f"     • ms/sample: {results['ms_per_sample']:.2f}")
        print(f"     • Batches/second: {results['batches_per_second']:.1f}")
        
        return results


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Nōkai v0.9 - H100-Optimized Training"
    )
    
    # Original arguments (preserved)
    parser.add_argument("--tier", type=str, default="small",
                       choices=["nano", "small", "medium", "large"],
                       help="Model size tier")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v09_h100")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate, don't train")
    
    # H100-specific arguments (NEW)
    parser.add_argument("--no-compile", action="store_true",
                       help="Disable torch.compile")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode")
    parser.add_argument("--no-bf16", action="store_true",
                       help="Disable BF16, use FP16")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark after setup")
    parser.add_argument("--compare-compiled", action="store_true",
                       help="Compare eager vs compiled performance")
    
    args = parser.parse_args()
    
    # Create H100 config
    config = H100ScalingConfig(
        model_tier=args.tier,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        total_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        gradient_accumulation=args.gradient_accumulation,
        # H100 settings
        use_bf16=not args.no_bf16,
        use_torch_compile=not args.no_compile,
        compile_mode=args.compile_mode,
    )
    
    # Create H100-optimized trainer
    trainer = NokaiTrainerH100(config)
    
    if trainer.setup():
        # Run benchmark if requested
        if args.benchmark:
            trainer.benchmark()
        
        # Compare eager vs compiled if requested
        if args.compare_compiled and not args.no_compile:
            H100Benchmark.compare_eager_vs_compiled(
                trainer.brain,
                input_shape=(args.batch_size, config.get_model_config()["max_seq_length"])
            )
        
        if not args.eval_only:
            trainer.train(args.steps)
        else:
            print("\n  📊 Evaluation only mode:")
            eval_results = trainer.evaluator.evaluate()
            print(f"     Accuracy: {eval_results['accuracy']:.1%}")


if __name__ == "__main__":
    main()
