import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Optional
import json

from src.training.metrics import SegmentationMetrics

class Trainer:
    """
    Training manager for segmentation model
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: str = 'cuda',
                 num_classes: int = 7,
                 ignore_index: int = 6,
                 checkpoint_dir: str = 'outputs/checkpoints',
                 use_wandb: bool = True):
        """
        Args:
            model: Segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device for training
            num_classes: Number of classes
            ignore_index: Class to ignore
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        # Initialize metrics
        self.train_metrics = SegmentationMetrics(num_classes, ignore_index, device)
        self.val_metrics = SegmentationMetrics(num_classes, ignore_index, device)

        # Tracking
        self.current_epoch = 0
        self.best_miou = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_miou': [],
            'val_miou': [],
            'lr': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs.detach(), masks)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.train_metrics.compute()
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Update metrics
            total_loss += loss.item()
            self.val_metrics.update(outputs, masks)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.compute()
        metrics['loss'] = avg_loss

        return metrics

    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Full training loop

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 60)

        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_miou'].append(train_metrics['mIoU'])
            self.history['val_miou'].append(val_metrics['mIoU'])
            self.history['lr'].append(current_lr)

            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train mIoU: {train_metrics['mIoU']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val mIoU:   {val_metrics['mIoU']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/mIoU': train_metrics['mIoU'],
                    'val/loss': val_metrics['loss'],
                    'val/mIoU': val_metrics['mIoU'],
                    'lr': current_lr
                }
                # Add per-class IoU
                for key, value in val_metrics.items():
                    if key.startswith('IoU_'):
                        log_dict[f'val/{key}'] = value

                wandb.log(log_dict)

            # Save checkpoint if best
            if val_metrics['mIoU'] > self.best_miou:
                self.best_miou = val_metrics['mIoU']
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best mIoU: {self.best_miou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Save regular checkpoint every 10 epochs
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            print("-" * 60)

        # Save final checkpoint and history
        self.save_checkpoint('final_model.pth')
        self.save_history()

        print(f"\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_miou = checkpoint['best_miou']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_history(self):
        """Save training history to JSON"""
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
