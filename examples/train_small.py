"""
Example: Training Nōkai on a small corpus

This demonstrates how to train Nōkai with minimal resources.
"""

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from nokai import NokaiConfig, NokaiModel
from nokai.training import train_nokai


def create_tokenizer(texts: list, vocab_size: int = 8000) -> Tokenizer:
    """Create a simple BPE tokenizer."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def main():
    # Sample training data
    texts = [
        "The brain is an amazing organ that processes information in complex ways.",
        "Neural networks are inspired by biological neurons in the human brain.",
        "Machine learning algorithms can learn patterns from data.",
        "Artificial intelligence is transforming many industries.",
        "Deep learning has achieved remarkable results in recent years.",
        # Add more text here...
    ] * 100  # Repeat for more training data
    
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(texts, vocab_size=8000)
    
    print("Configuring Nōkai...")
    # Use nano config for quick testing
    config = NokaiConfig.nano()
    config.vocab_size = tokenizer.get_vocab_size()
    config.max_sequence_length = 128
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Estimated parameters: {config.estimate_parameters():,}")
    print(f"Estimated VRAM: {config.estimate_vram_mb():.1f} MB")
    
    print("Training Nōkai...")
    model = train_nokai(
        config=config,
        train_texts=texts,
        tokenizer=tokenizer,
        num_epochs=5,
        batch_size=4,
    )
    
    print("Testing generation...")
    model.eval()
    
    prompt = "The brain"
    prompt_tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_tokens], device=config.device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
        )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
