"""
Efficient Zero-Copy Data Pipeline

This module implements memory-efficient data loading:
- Memory-mapped files (data stays on disk)
- Streaming from disk (constant RAM usage)
- Prefetching for speed
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, Iterator, Generator
from pathlib import Path
import struct


class MemoryMappedTokenDataset(Dataset):
    """
    Dataset using memory-mapped files for zero-copy data access.
    
    Data stays on disk, only requested samples are loaded into RAM.
    This allows training on datasets larger than RAM.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 512,
        dtype: np.dtype = np.int32,
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Get file size to calculate number of tokens
        file_size = os.path.getsize(data_path)
        self.num_tokens = file_size // self.itemsize
        self.num_samples = max(0, (self.num_tokens - 1) // sequence_length)
        
        # Memory-map the file (read-only)
        self._mmap = np.memmap(data_path, dtype=dtype, mode='r')
        
        print(f"📂 Loaded memory-mapped dataset:")
        print(f"   Tokens: {self.num_tokens:,}")
        print(f"   Samples: {self.num_samples:,}")
        print(f"   RAM usage: ~0 MB (memory-mapped)")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        start = idx * self.sequence_length
        end = start + self.sequence_length + 1
        
        # Read directly from mmap (no copy until .copy())
        tokens = self._mmap[start:end].astype(np.int64)
        
        return {
            'input_ids': torch.from_numpy(tokens[:-1].copy()),
            'labels': torch.from_numpy(tokens[1:].copy()),
        }
    
    def __del__(self):
        if hasattr(self, '_mmap'):
            del self._mmap


class StreamingTokenDataset(IterableDataset):
    """
    Streaming dataset that reads data in chunks.
    
    Perfect for very large datasets - uses constant memory
    regardless of dataset size.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 512,
        chunk_size: int = 10000,
        shuffle_chunks: bool = True,
        dtype: np.dtype = np.int32,
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.dtype = dtype
        
        # Calculate total tokens
        file_size = os.path.getsize(data_path)
        self.num_tokens = file_size // np.dtype(dtype).itemsize
        self.num_samples = max(0, (self.num_tokens - 1) // sequence_length)
    
    def __iter__(self) -> Iterator[dict]:
        # Open file for streaming
        mmap = np.memmap(self.data_path, dtype=self.dtype, mode='r')
        
        # Generate chunk indices
        num_chunks = (self.num_samples + self.chunk_size - 1) // self.chunk_size
        chunk_indices = list(range(num_chunks))
        
        if self.shuffle_chunks:
            np.random.shuffle(chunk_indices)
        
        for chunk_idx in chunk_indices:
            start_sample = chunk_idx * self.chunk_size
            end_sample = min(start_sample + self.chunk_size, self.num_samples)
            
            # Get indices for this chunk
            sample_indices = list(range(start_sample, end_sample))
            if self.shuffle_chunks:
                np.random.shuffle(sample_indices)
            
            for idx in sample_indices:
                start = idx * self.sequence_length
                end = start + self.sequence_length + 1
                
                tokens = mmap[start:end].astype(np.int64)
                
                yield {
                    'input_ids': torch.from_numpy(tokens[:-1].copy()),
                    'labels': torch.from_numpy(tokens[1:].copy()),
                }
        
        del mmap


def preprocess_to_binary(
    texts: list,
    tokenizer,
    output_path: str,
    show_progress: bool = True,
) -> int:
    """
    Preprocess text corpus into binary token file.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer with encode method
        output_path: Path to save binary file
        
    Returns:
        Total number of tokens
    """
    from tqdm import tqdm
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # First pass: count tokens to allocate array
    total_tokens = 0
    iterator = tqdm(texts, desc="Counting tokens") if show_progress else texts
    for text in iterator:
        total_tokens += len(tokenizer.encode(text).ids) + 1  # +1 for EOS
    
    print(f"📊 Total tokens: {total_tokens:,}")
    
    # Create memory-mapped file for writing
    mmap = np.memmap(output_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
    
    # Second pass: write tokens
    eos_id = tokenizer.token_to_id("[EOS]") or 0
    current_pos = 0
    
    iterator = tqdm(texts, desc="Writing tokens") if show_progress else texts
    for text in iterator:
        tokens = tokenizer.encode(text).ids
        n = len(tokens)
        mmap[current_pos:current_pos + n] = tokens
        current_pos += n
        mmap[current_pos] = eos_id
        current_pos += 1
    
    # Flush to disk
    mmap.flush()
    del mmap
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved to {output_path} ({file_size_mb:.1f} MB)")
    
    return total_tokens


def create_efficient_dataloader(
    data_path: str,
    sequence_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    streaming: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create an efficient DataLoader with minimal memory usage.
    """
    if streaming:
        dataset = StreamingTokenDataset(
            data_path,
            sequence_length=sequence_length,
        )
    else:
        dataset = MemoryMappedTokenDataset(
            data_path,
            sequence_length=sequence_length,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not streaming,  # Streaming handles its own shuffling
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
