#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nōkai Data Downloader - TinyStories Dataset

Downloads and prepares the TinyStories dataset for training.
TinyStories is a synthetic dataset of short stories written for 
children, perfect for training small language models.

Reference:
    Eldan, Ronen and Li, Yuanzhi. "TinyStories: How Small Can 
    Language Models Be and Still Speak Coherent English?" (2023)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --split 0.05  # 5% of data
    python scripts/download_data.py --split 0.01 --output data/tiny_1pct.txt
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_tinystories(
    output_path: str = "data/tinystories.txt",
    split_fraction: float = 0.05,
    min_length: int = 100,
    max_samples: int = None,
    seed: int = 42,
):
    """
    Download and save TinyStories dataset.
    
    Args:
        output_path: Where to save the text file
        split_fraction: Fraction of training data to use (0.01 = 1%)
        min_length: Minimum story length in characters
        max_samples: Maximum number of samples (None = use split_fraction)
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("📚 NŌKAI DATA DOWNLOADER - TinyStories")
    print("=" * 60)
    
    # Check for datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n❌ Error: 'datasets' library required.")
        print("   Install with: pip install datasets")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # =====================================
    # DOWNLOAD DATASET
    # =====================================
    print(f"\n[1/4] Downloading TinyStories from HuggingFace...")
    print(f"      This may take a few minutes on first run...")
    
    try:
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTrying alternative method...")
        try:
            dataset = load_dataset(
                "roneneldan/TinyStories",
                split="train[:10%]",  # Load only 10% to reduce memory
            )
        except Exception as e2:
            print(f"❌ Failed: {e2}")
            sys.exit(1)
    
    total_samples = len(dataset)
    print(f"      Total samples available: {total_samples:,}")
    
    # =====================================
    # SELECT SUBSET
    # =====================================
    print(f"\n[2/4] Selecting subset...")
    
    if max_samples is not None:
        num_samples = min(max_samples, total_samples)
    else:
        num_samples = int(total_samples * split_fraction)
    
    num_samples = max(1000, num_samples)  # At least 1000 samples
    print(f"      Selecting {num_samples:,} samples ({100 * num_samples / total_samples:.1f}%)")
    
    # Shuffle and select
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(num_samples, total_samples)))
    
    # =====================================
    # FILTER AND CLEAN
    # =====================================
    print(f"\n[3/4] Filtering and cleaning...")
    
    stories = []
    skipped = 0
    
    for item in dataset:
        text = item.get('text', '')
        
        # Basic cleaning
        text = text.strip()
        
        # Skip too short
        if len(text) < min_length:
            skipped += 1
            continue
        
        # Skip if no actual content
        if text.count(' ') < 10:
            skipped += 1
            continue
        
        stories.append(text)
    
    print(f"      Valid stories: {len(stories):,}")
    print(f"      Skipped (too short): {skipped:,}")
    
    if len(stories) < 100:
        print("\n⚠️  Warning: Very few stories selected. Consider increasing --split")
    
    # =====================================
    # SAVE TO FILE
    # =====================================
    print(f"\n[4/4] Saving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for story in stories:
            # Write each story on its own line, with double newline separator
            f.write(story)
            f.write("\n\n")
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    
    # =====================================
    # SUMMARY
    # =====================================
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"   Output file: {output_path}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Total stories: {len(stories):,}")
    print(f"   Average length: {sum(len(s) for s in stories) / len(stories):.0f} chars")
    print("=" * 60)
    
    # Calculate some stats for tokenizer
    total_chars = sum(len(s) for s in stories)
    unique_words = set()
    for story in stories[:1000]:  # Sample for word count
        unique_words.update(story.lower().split())
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Unique words (sample): ~{len(unique_words):,}")
    print(f"   Estimated BPE vocab: {min(32000, len(unique_words) * 3):,}+")
    
    print(f"\n🚀 Next step: Train with this data:")
    print(f"   python scripts/train_cognitive_v2.py \\")
    print(f"       --data_file {output_path} \\")
    print(f"       --preset nano \\")
    print(f"       --epochs 5")
    
    return str(output_path), len(stories)


def main():
    parser = argparse.ArgumentParser(
        description="Download TinyStories dataset for Nōkai training"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/tinystories.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--split", "-s",
        type=float,
        default=0.05,
        help="Fraction of data to download (0.01=1%%, 0.05=5%%, 0.1=10%%)"
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples (overrides --split)"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum story length in characters"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    download_tinystories(
        output_path=args.output,
        split_fraction=args.split,
        min_length=args.min_length,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
