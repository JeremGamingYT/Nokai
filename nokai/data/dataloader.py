"""
Efficient Data Loading - Memory-mapped streaming

Loads large datasets from disk without consuming RAM.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Iterator, List
import mmap


class MemoryMappedDataset(Dataset):
    """
    Dataset that streams from memory-mapped files.
    
    Data stays on disk, only requested samples are loaded.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 512,
        dtype: np.dtype = np.int32,
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.dtype = dtype
        
        # Open memory-mapped file
        if os.path.exists(data_path):
            self.data = np.memmap(data_path, dtype=dtype, mode='r')
            self.num_tokens = len(self.data)
            self.num_samples = self.num_tokens // sequence_length
        else:
            self.data = None
            self.num_tokens = 0
            self.num_samples = 0
    
    def __len__(self) -> int:
        return max(0, self.num_samples - 1)
    
    def __getitem__(self, idx: int) -> dict:
        if self.data is None:
            return {'input_ids': torch.zeros(self.sequence_length, dtype=torch.long)}
        
        start = idx * self.sequence_length
        end = start + self.sequence_length + 1
        
        tokens = self.data[start:end].astype(np.int64)
        
        return {
            'input_ids': torch.from_numpy(tokens[:-1]).long(),
            'labels': torch.from_numpy(tokens[1:]).long(),
        }


class StreamingDataset(Dataset):
    """
    Streaming dataset that processes text on-the-fly.
    
    For smaller datasets or when preprocessing is not done.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        sequence_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Pre-tokenize for efficiency
        self.tokens = []
        for text in texts:
            self.tokens.extend(tokenizer.encode(text))
        
        self.num_samples = len(self.tokens) // sequence_length
    
    def __len__(self) -> int:
        return max(0, self.num_samples - 1)
    
    def __getitem__(self, idx: int) -> dict:
        start = idx * self.sequence_length
        end = start + self.sequence_length + 1
        
        tokens = self.tokens[start:end]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Create an optimized DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def preprocess_corpus(
    input_path: str,
    output_path: str,
    tokenizer,
    chunk_size: int = 10000,
) -> int:
    """
    Preprocess text corpus into memory-mapped format.
    
    Returns total number of tokens.
    """
    all_tokens = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                text = ' '.join(chunk)
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                chunk = []
        
        if chunk:
            text = ' '.join(chunk)
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
    
    # Save as memory-mapped file
    tokens_array = np.array(all_tokens, dtype=np.int32)
    mmap_array = np.memmap(output_path, dtype=np.int32, mode='w+', shape=tokens_array.shape)
    mmap_array[:] = tokens_array
    mmap_array.flush()
    
    return len(all_tokens)
