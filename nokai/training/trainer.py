"""
Nōkai Training Script

Combines gradient-based learning with Hebbian plasticity.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Optional, Dict
from tqdm import tqdm

from nokai.config import NokaiConfig
from nokai.model import NokaiModel
from nokai.data import create_dataloader, StreamingDataset


class NokaiTrainer:
    """
    Hybrid trainer combining:
    - Gradient descent for global optimization
    - Hebbian learning for local refinement
    - Memory consolidation during "sleep" phases
    """
    
    def __init__(
        self,
        model: NokaiModel,
        config: NokaiConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning.learning_rate,
            weight_decay=config.learning.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.memory_optimization.mixed_precision else None
        
        # Tracking
        self.step = 0
        self.epoch = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward with mixed precision
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = self.model(input_ids, labels=labels, store_memory=True)
            loss = outputs['loss']
        
        # Backward
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.learning.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.learning.gradient_clip
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.step += 1
        
        # Apply Hebbian updates periodically
        if self.step % 10 == 0 and self.config.learning.plasticity_modulation:
            self._apply_hebbian_updates()
        
        # Memory consolidation (sleep phase)
        if self.step % self.config.learning.consolidation_interval == 0:
            self.model.consolidate_memory()
        
        return {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def _apply_hebbian_updates(self):
        """Apply local Hebbian learning to cortical columns."""
        with torch.no_grad():
            for layer in self.model.cortex.layers:
                for column in layer.columns:
                    updates = column.get_hebbian_update(
                        learning_rate=self.config.learning.hebbian_lr
                    )
                    column.apply_hebbian_update(updates)
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.epoch = epoch
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['lr']:.2e}",
            })
        
        return {'epoch_loss': total_loss / num_batches}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'config': self.config.model_dump(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']


def train_nokai(
    config: NokaiConfig,
    train_texts: list,
    tokenizer,
    num_epochs: int = 10,
    batch_size: int = 8,
    checkpoint_dir: str = "checkpoints",
):
    """Main training function."""
    
    # Create model
    model = NokaiModel(config)
    
    # Create dataset
    dataset = StreamingDataset(
        train_texts,
        tokenizer,
        sequence_length=config.max_sequence_length,
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
    )
    
    # Create trainer
    trainer = NokaiTrainer(model, config, device=config.device)
    
    # Training loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}: Loss = {metrics['epoch_loss']:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(
            os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        )
    
    return model
