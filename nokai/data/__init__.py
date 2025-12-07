"""Data module for Nōkai with memory-efficient loading."""

from nokai.data.dataloader import (
    MemoryMappedTokenDataset,
    StreamingTokenDataset,
    preprocess_to_binary,
    create_efficient_dataloader,
)

# Legacy compatibility
class StreamingDataset(StreamingTokenDataset):
    pass

def create_dataloader(*args, **kwargs):
    """Legacy wrapper."""
    from torch.utils.data import DataLoader
    return DataLoader(*args, **kwargs)

__all__ = [
    "MemoryMappedTokenDataset",
    "StreamingTokenDataset",
    "StreamingDataset",
    "preprocess_to_binary",
    "create_efficient_dataloader",
    "create_dataloader",
]
