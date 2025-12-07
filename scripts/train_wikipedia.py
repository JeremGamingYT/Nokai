"""
Nōkai Training on Wikipedia

This script:
1. Downloads Wikipedia data (French or English)
2. Preprocesses and tokenizes
3. Trains Nōkai with hybrid learning
"""

import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import normalizers

from nokai import NokaiConfig, NokaiModel
from nokai.training import NokaiTrainer
from nokai.data import StreamingDataset, create_dataloader


def download_wikipedia(
    language: str = "en",
    num_samples: int = 100_000,
    cache_dir: str = "./data/cache",
) -> list:
    """
    Download Wikipedia articles.
    
    Args:
        language: 'en' for English, 'fr' for French, etc.
        num_samples: Number of articles to download
        cache_dir: Cache directory for datasets
        
    Returns:
        List of article texts
    """
    print(f"📥 Downloading Wikipedia ({language})...")
    print(f"   This may take a while on first run...")
    
    # Use the new Wikimedia Wikipedia dataset (Parquet format)
    # Available: 20231101 snapshot
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{language}",
            split="train",
            cache_dir=cache_dir,
            streaming=True,  # Stream to avoid downloading everything
        )
        
        # Take samples from streaming dataset
        texts = []
        for i, article in enumerate(tqdm(dataset, desc="Downloading", total=num_samples)):
            if i >= num_samples:
                break
            text = article.get("text", "")
            if len(text) > 100:  # Skip very short articles
                texts.append(text)
        
        print(f"✓ Downloaded {len(texts):,} articles")
        return texts
        
    except Exception as e:
        print(f"⚠️ Wikipedia download failed: {e}")
        print("   Using fallback: simple text generation for testing...")
        
        # Fallback: generate simple training data
        sample_texts = [
            "The human brain is a complex organ responsible for controlling all functions of the body.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
            "Deep learning uses artificial neural networks with multiple layers to model complex patterns.",
            "The neocortex is the part of the brain responsible for higher-order functions like language and reasoning.",
            "Synapses are the connections between neurons where information is transmitted.",
            "Memory consolidation occurs during sleep when the brain strengthens neural connections.",
            "The hippocampus plays a crucial role in forming new memories and spatial navigation.",
            "Attention mechanisms in the brain help focus on relevant information while filtering out noise.",
            "Neuroplasticity refers to the brain's ability to reorganize itself by forming new neural connections.",
        ] * (num_samples // 10 + 1)
        
        return sample_texts[:num_samples]


def create_tokenizer_from_texts(
    texts: list,
    vocab_size: int = 32000,
    save_path: str = "./data/tokenizer.json",
) -> Tokenizer:
    """
    Train a BPE tokenizer on the corpus.
    """
    print(f"🔤 Training tokenizer (vocab_size={vocab_size})...")
    
    # Create tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Normalizer
    tokenizer.normalizer = normalizers.Sequence([
        NFD(),
        Lowercase(),
        StripAccents(),
    ])
    
    # Pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]"],
        min_frequency=2,
        show_progress=True,
    )
    
    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"✓ Tokenizer saved to {save_path}")
    
    return tokenizer


def prepare_training_data(
    texts: list,
    tokenizer: Tokenizer,
    sequence_length: int = 512,
    save_path: str = "./data/train_tokens.bin",
) -> str:
    """
    Tokenize all texts and save as binary file for memory-mapping.
    """
    import numpy as np
    
    print(f"📝 Tokenizing corpus...")
    
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.token_to_id("[EOS]"))  # Add EOS
    
    print(f"✓ Total tokens: {len(all_tokens):,}")
    
    # Save as memory-mapped file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokens_array = np.array(all_tokens, dtype=np.int32)
    tokens_array.tofile(save_path)
    
    print(f"✓ Saved to {save_path} ({os.path.getsize(save_path) / 1e6:.1f} MB)")
    
    return save_path


def train_nokai(
    config: NokaiConfig,
    dataset: StreamingDataset,
    num_epochs: int = 10,
    batch_size: int = 8,
    save_dir: str = "./checkpoints",
    log_interval: int = 100,
):
    """
    Train Nōkai model.
    """
    print(f"\n🧠 Starting Nōkai Training")
    print(f"   Config: {config.num_columns} columns, {config.column_config.num_neurons} neurons/col")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {config.device}")
    
    # Create model
    model = NokaiModel(config)
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Windows compatibility
        shuffle=True,
    )
    
    # Create trainer
    trainer = NokaiTrainer(model, config, device=config.device)
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Training")
        for batch in pbar:
            metrics = trainer.train_step(batch)
            
            epoch_loss += metrics['loss']
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['lr']:.2e}",
            })
            
            # Log
            if global_step % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                print(f"\n   Step {global_step}: loss={avg_loss:.4f}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\n   Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"nokai_epoch_{epoch + 1}.pt")
        trainer.save_checkpoint(checkpoint_path)
        print(f"   ✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(save_dir, "nokai_best.pt")
            trainer.save_checkpoint(best_path)
            print(f"   ⭐ New best model saved!")
    
    print(f"\n🎉 Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints saved in: {save_dir}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Nōkai on Wikipedia")
    
    # Data arguments
    parser.add_argument("--language", type=str, default="en", 
                        help="Wikipedia language (en, fr, de, etc.)")
    parser.add_argument("--num-articles", type=int, default=10000,
                        help="Number of Wikipedia articles to use")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size for tokenizer")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length for training")
    
    # Model arguments
    parser.add_argument("--model-size", type=str, default="mini",
                        choices=["nano", "micro", "mini", "base", "large"],
                        help="Model size preset")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    
    # Flags
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use cached)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_path = data_dir / "tokenizer.json"
    tokens_path = data_dir / "train_tokens.bin"
    
    # Step 1: Download Wikipedia
    if not args.skip_download or not tokens_path.exists():
        texts = download_wikipedia(
            language=args.language,
            num_samples=args.num_articles,
            cache_dir=str(data_dir / "cache"),
        )
        
        # Step 2: Create tokenizer
        tokenizer = create_tokenizer_from_texts(
            texts,
            vocab_size=args.vocab_size,
            save_path=str(tokenizer_path),
        )
        
        # Step 3: Prepare training data
        prepare_training_data(
            texts,
            tokenizer,
            sequence_length=args.seq_length,
            save_path=str(tokens_path),
        )
    else:
        print("📂 Using cached data...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Step 4: Create configuration
    config_factory = {
        "nano": NokaiConfig.nano,
        "micro": NokaiConfig.micro,
        "mini": NokaiConfig.mini,
        "base": NokaiConfig.base,
        "large": NokaiConfig.large,
    }
    
    config = config_factory[args.model_size]()
    config.vocab_size = tokenizer.get_vocab_size()
    config.max_sequence_length = args.seq_length
    config.learning.learning_rate = args.lr
    
    # Device
    if args.cpu or not torch.cuda.is_available():
        config.device = "cpu"
        print("⚠️  Training on CPU (will be slow)")
    else:
        config.device = "cuda"
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name()}")
    
    print(f"\n📊 Model Configuration:")
    print(f"   Size: {args.model_size}")
    print(f"   Parameters: ~{config.estimate_parameters():,}")
    print(f"   VRAM: ~{config.estimate_vram_mb():.0f} MB")
    
    # Step 5: Create dataset
    # Load tokenized data
    import numpy as np
    tokens = np.fromfile(str(tokens_path), dtype=np.int32)
    
    # Create simple dataset
    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, seq_length):
            self.tokens = tokens
            self.seq_length = seq_length
            self.num_samples = len(tokens) // seq_length - 1
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            start = idx * self.seq_length
            end = start + self.seq_length + 1
            chunk = self.tokens[start:end]
            return {
                'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                'labels': torch.tensor(chunk[1:], dtype=torch.long),
            }
    
    dataset = TokenDataset(tokens, args.seq_length)
    print(f"   Training samples: {len(dataset):,}")
    
    # Step 6: Train!
    model = train_nokai(
        config=config,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )
    
    # Step 7: Test generation
    print("\n📝 Testing generation...")
    model.eval()
    
    prompt = "The brain"
    prompt_tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_tokens], device=config.device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
        )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
