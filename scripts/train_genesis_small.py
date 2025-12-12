"""
GENESIS Small Training - Optimized for 6GB VRAM

=============================================================================
TRAINING GOALS
=============================================================================

1. Validate GENESIS architecture can learn
2. Test one-shot learning capability
3. Measure memory efficiency
4. Compare with traditional approach

Target: 6GB VRAM, 10GB RAM
Model: ~50M parameters (ternary = ~6.25MB storage!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import gc
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class GenesisTrainConfig:
    """Training configuration for 6GB VRAM."""

    # Model size (conservative for 6GB)
    vocab_size: int = 8192
    embedding_dim: int = 256
    num_layers: int = 4
    num_neurons_per_layer: int = 512

    # Lightweight context modeling
    use_context_rnn: bool = True
    rnn_layers: int = 1
    
    # Training
    batch_size: int = 8
    seq_length: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 3
    gradient_accumulation: int = 4

    # Phase-specific epochs
    completion_epochs: int = 10
    dialogue_epochs: int = 20
    
    # GENESIS specific
    use_ternary: bool = True
    target_sparsity: float = 0.9
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    empty_cache_frequency: int = 10
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


class TinyTextDataset(Dataset):
    """Simple text dataset for testing."""
    
    def __init__(self, texts: list, tokenizer, seq_length: int = 128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize all texts
        self.tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.tokens.extend(tokens)
        
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        
        # Number of samples
        self.num_samples = max(1, (len(self.tokens) - 1) // seq_length)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        
        if end > len(self.tokens):
            # Pad if necessary
            chunk = self.tokens[start:]
            padding = torch.zeros(end - len(self.tokens), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        else:
            chunk = self.tokens[start:end]
        
        return chunk[:-1], chunk[1:]  # input, target


class SimpleTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 8192):
        # Basic ASCII + special tokens
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Special tokens
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Build vocab
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<UNK>'] = 1
        self.char_to_id['<BOS>'] = 2
        self.char_to_id['<EOS>'] = 3
        
        # ASCII printable
        for i, c in enumerate(range(32, 127)):
            self.char_to_id[chr(c)] = i + 4
            self.id_to_char[i + 4] = chr(c)
    
    def encode(self, text: str) -> list:
        return [self.char_to_id.get(c, self.unk_id) for c in text]
    
    def decode(self, ids: list) -> str:
        decoded = []
        for i in ids:
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            decoded.append(self.id_to_char.get(i, '?'))
        return ''.join(decoded)


class GenesisBlock(nn.Module):
    """Single GENESIS block with local learning."""
    
    def __init__(
        self,
        dim: int,
        num_neurons: int,
        use_ternary: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_neurons = num_neurons
        
        # Import GENESIS components
        from nokai.genesis.rnu import RichNeuronUnit, RNUConfig
        from nokai.genesis.learning import GenesisLearning
        
        # RNU layer
        config = RNUConfig(
            stochastic=True,
            ternary_output=False,  # Use continuous for training
            target_sparsity=0.9,
        )
        
        self.rnu = RichNeuronUnit(
            num_neurons=num_neurons,
            input_dim=dim,
            config=config,
            use_ternary_weights=use_ternary,
        )
        
        # Output projection
        if use_ternary:
            from nokai.genesis.ternary import TernaryLinear
            self.output_proj = TernaryLinear(num_neurons, dim)
        else:
            self.output_proj = nn.Linear(num_neurons, dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Local learner
        self.learner = GenesisLearning(dim, num_neurons)
        
        # Store activations for local learning
        self.last_input = None
        self.last_hidden = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Store input for local learning
        if self.training:
            self.last_input = x.detach()
        
        # RNU forward
        # Flatten batch and sequence
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        
        hidden, _ = self.rnu(x_flat)
        
        if self.training:
            self.last_hidden = hidden.detach()
        
        # Project back
        out = self.output_proj(hidden)
        out = out.view(batch, seq, dim)
        
        # Residual + norm
        out = self.norm(residual + out)
        
        return out
    
    def local_learn(self, dopamine: float = 1.0, acetylcholine: float = 0.7) -> Dict:
        """Apply local learning."""
        if self.last_input is None:
            return {}
        
        # Get weight
        weight = self.rnu.synapses.weight
        
        success, info = self.learner.apply_local_update(
            weight=weight,
            pre=self.last_input.view(-1, self.dim),
            post=self.last_hidden,
            dopamine=dopamine,
            acetylcholine=acetylcholine,
        )
        
        return info


class GenesisLM(nn.Module):
    """
    GENESIS Language Model - Optimized for low VRAM.
    
    Architecture:
    - Ternary embeddings
    - RNU blocks with local learning
    - Ternary output projection
    """
    
    def __init__(self, config: GenesisTrainConfig):
        super().__init__()
        
        self.config = config
        
        # Embedding (ternary optional)
        if config.use_ternary:
            from nokai.genesis.ternary import TernaryEmbedding
            self.embedding = TernaryEmbedding(
                config.vocab_size,
                config.embedding_dim,
                sparsity=0.8,
            )
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Positional encoding (simple learned)
        self.pos_embedding = nn.Embedding(config.seq_length, config.embedding_dim)

        # Simple causal context mixer (GRU)
        self.context_rnn = None
        if config.use_context_rnn:
            self.context_rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.embedding_dim,
                num_layers=config.rnn_layers,
                batch_first=True,
            )

        # GENESIS blocks
        self.blocks = nn.ModuleList([
            GenesisBlock(
                dim=config.embedding_dim,
                num_neurons=config.num_neurons_per_layer,
                use_ternary=config.use_ternary,
            )
            for _ in range(config.num_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        
        if config.use_ternary:
            from nokai.genesis.ternary import TernaryLinear
            self.output_head = TernaryLinear(config.embedding_dim, config.vocab_size)
        else:
            self.output_head = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # Count parameters
        self._count_params()
    
    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if self.config.use_ternary:
            # Ternary params use 2 bits instead of 32
            storage_mb = total * 2 / 8 / 1024 / 1024
        else:
            storage_mb = total * 4 / 1024 / 1024
        
        print(f"Model parameters: {total:,} ({trainable:,} trainable)")
        print(f"Estimated storage: {storage_mb:.2f} MB")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        
        # Add position
        pos = torch.arange(seq, device=input_ids.device)
        x = x + self.pos_embedding(pos)

        # Share context across the sequence
        if self.context_rnn is not None:
            x, _ = self.context_rnn(x)

        # Blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.output_norm(x)
        logits = self.output_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,  # Ignore padding
            )
        
        return logits, loss
    
    def apply_local_learning(self, dopamine: float = 1.0) -> Dict:
        """Apply local learning to all blocks."""
        total_info = {}
        
        for i, block in enumerate(self.blocks):
            info = block.local_learn(dopamine=dopamine)
            if info:
                total_info[f'block_{i}'] = info
        
        return total_info
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 20,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        generated = prompt_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Only use last seq_length tokens
                context = generated[:, -self.config.seq_length:]
                
                logits, _ = self(context)
                next_logits = logits[:, -1, :]

                # Avoid sampling padding/unknown control tokens
                next_logits = next_logits.clone()
                next_logits[:, 0] = -float('inf')  # PAD
                next_logits[:, 1] = -float('inf')  # UNK
                next_logits[:, 2] = -float('inf')  # BOS

                # Sampling controls
                scaled_logits = next_logits / max(temperature, 1e-4)

                if top_k is not None and top_k > 0:
                    top_k = min(top_k, scaled_logits.size(-1))
                    top_values, _ = torch.topk(scaled_logits, top_k)
                    cutoff = top_values[:, -1].unsqueeze(-1)
                    scaled_logits = torch.where(
                        scaled_logits < cutoff,
                        torch.full_like(scaled_logits, -float('inf')),
                        scaled_logits,
                    )

                probs = F.softmax(scaled_logits, dim=-1)

                if greedy:
                    next_token = probs.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, 1)

                generated = torch.cat([generated, next_token], dim=1)

        return generated


def build_completion_corpus():
    """Structured sentences for the completion phase."""

    texts = [
        "Complete the sentence: The quick brown fox jumps over the lazy dog.",
        "Finish this idea: Clear writing uses short sentences and precise words.",
        "Complete the phrase: After the rain, the street shines with small puddles.",
        "Continue the thought: Good explanations start simple and add detail slowly.",
        "Complete: The apple on the table is blue and glossy.",
        "Phrase to finish: A helpful model answers briefly and stays on topic.",
        "Complete the story: She opened the book and found a map to the hidden garden.",
        "Finish: The brain learns patterns by seeing many well formed sentences.",
        "Complete this line: Practice every day to write better and think clearly.",
        "Continue the idea: When light hits water, it bends and makes colors.",
        "Complete this example: The sentence begins with a subject and ends with a clear verb.",
        "Finish the notion: Curiosity makes people ask questions and search for answers.",
        "Complete the color memory: The apple is blue and tastes sweet.",
        "Complete the dialogue tag: The assistant replies with calm and clarity.",
        "Finish this observation: Clean text has proper spacing and punctuation.",
        "Complete the reminder: Use simple words to keep the style stable.",
    ]

    return texts * 20


def build_dialogue_corpus():
    """User/assistant patterns for conversational alignment."""

    dialogues = [
        "user: hello, can you complete this sentence about the sky? assistant: the sky is clear, blue, and calm this morning.",
        "user: finish the phrase the apple is assistant: the apple is blue, sweet, and ready to eat.",
        "user: how should i respond politely? assistant: respond briefly, stay kind, and address the question directly.",
        "user: give me a clean answer to what is dopamine? assistant: dopamine is a neurotransmitter that signals motivation and reward.",
        "user: help me write correctly in simple english assistant: keep sentences short, use clear subjects, and avoid filler words.",
        "user: continue this line learning happens when assistant: learning happens when practice, feedback, and curiosity work together.",
        "user: act as a helpful assistant assistant: sure, i will answer in full sentences and stay respectful.",
        "user: how do you describe a blue apple assistant: a blue apple looks unusual, shiny, and memorable.",
        "user: can you rewrite this politely thanks assistant: thank you for asking, here is a concise and polite reply.",
        "user: what should an assistant say if it does not know assistant: it should admit it, stay calm, and ask for more details.",
        "user: help me end this phrase the model should assistant: the model should stay coherent, safe, and easy to read.",
        "user: what is the point of training phases assistant: first learn to complete sentences, then learn to reply as an assistant.",
    ]

    return dialogues * 15


def create_dataloader(texts, tokenizer, config, device):
    dataset = TinyTextDataset(texts, tokenizer, config.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )

    return dataloader, dataset


def run_generation_tests(
    model,
    tokenizer,
    device,
    prompts,
    label: str,
    max_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 20,
    greedy: bool = True,
):
    print(f"\nTest Generation - {label}")
    model.eval()

    for prompt in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"  {label}: '{prompt}' -> '{generated_text}'")

    model.train()


def run_training_phase(
    model,
    tokenizer,
    optimizer,
    dataloader,
    device,
    config,
    phase_name: str,
    num_epochs: int,
    global_step: int,
    start_time: float,
    test_prompts=None,
    max_tokens: int = 40,
):
    if test_prompts is None:
        test_prompts = []

    last_epoch_loss = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits, loss = model(input_ids, targets)

            loss = loss / config.gradient_accumulation
            loss.backward()

            dopamine = max(0.3, min(1.0, 1.0 - loss.item()))
            model.apply_local_learning(dopamine=dopamine)

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation
            epoch_steps += 1
            global_step += 1

            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time

                print(
                    f"[{phase_name}] Epoch {epoch+1}/{num_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"DA: {dopamine:.2f} | "
                    f"Time: {elapsed:.1f}s"
                )

                if device.type == 'cuda':
                    print(f"  GPU Mem: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

            if global_step % config.empty_cache_frequency == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        last_epoch_loss = epoch_loss / max(1, epoch_steps)
        print(f"\n--- {phase_name} Epoch {epoch+1}/{num_epochs} Complete ---")
        print(f"Average Loss: {last_epoch_loss:.4f}")

        if test_prompts:
            run_generation_tests(
                model,
                tokenizer,
                device,
                test_prompts,
                label=f"{phase_name} preview",
                max_tokens=max_tokens,
            )

    return global_step, last_epoch_loss


def train():
    """Main training loop with phased corpora."""

    print("=" * 60)
    print("  GENESIS Training - 6GB VRAM Optimized")
    print("=" * 60)

    config = GenesisTrainConfig(
        vocab_size=256,  # Character level
        embedding_dim=256,
        num_layers=4,
        num_neurons_per_layer=512,
        batch_size=4,
        seq_length=64,
        completion_epochs=12,
        dialogue_epochs=18,
        use_ternary=True,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    tokenizer = SimpleTokenizer(config.vocab_size)

    phases = [
        {
            'name': 'completion',
            'texts': build_completion_corpus(),
            'epochs': config.completion_epochs,
            'test_prompts': [
                "Complete the phrase: the apple is",
                "Finish this idea: clear writing starts with",
            ],
            'max_tokens': 40,
        },
        {
            'name': 'dialogue',
            'texts': build_dialogue_corpus(),
            'epochs': config.dialogue_epochs,
            'test_prompts': [
                "user: can you complete this sentence about clarity? assistant:",
                "user: describe a blue apple assistant:",
            ],
            'max_tokens': 50,
        },
    ]

    print("\nCreating model...")
    model = GenesisLM(config).to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        print(f"GPU Memory after model: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 60)
    print("  Starting Training (two phases)")
    print("=" * 60)

    global_step = 0
    start_time = time.time()
    final_loss = 0.0

    for phase in phases:
        dataloader, dataset = create_dataloader(phase['texts'], tokenizer, config, device)
        print(
            f"\n== Phase: {phase['name']} | Samples: {len(dataset)} | Epochs: {phase['epochs']} =="
        )
        print(f"Steps per epoch: {len(dataloader)}")

        global_step, final_loss = run_training_phase(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            config=config,
            phase_name=phase['name'],
            num_epochs=phase['epochs'],
            global_step=global_step,
            start_time=start_time,
            test_prompts=phase.get('test_prompts', []),
            max_tokens=phase.get('max_tokens', 40),
        )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Final loss: {final_loss:.4f}")

    if device.type == 'cuda':
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    save_path = project_root / "checkpoints" / "genesis_small.pt"
    save_path.parent.mkdir(exist_ok=True)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': config,
            'final_loss': final_loss,
        },
        save_path,
    )

    print(f"Model saved to: {save_path}")

    final_prompts = [
        "Complete the phrase: the apple is",
        "Finish this idea: good writing is",
        "user: finish the phrase the apple is assistant:",
        "user: how should an assistant respond politely? assistant:",
    ]

    run_generation_tests(
        model,
        tokenizer,
        device,
        final_prompts,
        label="final",
        max_tokens=50,
    )

    return model


if __name__ == "__main__":
    train()
