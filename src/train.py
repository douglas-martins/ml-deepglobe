import sys
import torch
import json
import wandb

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DeepGlobeDataset
from src.models.segmentation_model import create_model
from src.training.losses import CombinedLoss, compute_class_weights
from src.training.trainer import Trainer
from src.utils.torch_device import get_device

# Configuration
CONFIG = {
    # Data
    'data_dir': str(PROJECT_ROOT / 'data' / 'raw' / 'train'),
    'train_split': str(PROJECT_ROOT / 'data' / 'splits' / 'train_files.json'),
    'val_split': str(PROJECT_ROOT / 'data' / 'splits' / 'val_files.json'),
    'tile_size': 384, # 512
    'overlap': 64,

    # Model
    'architecture': 'unet',  # 'unet' or 'deeplabv3plus'
    'encoder': 'resnet34',   # 'resnet34', 'resnet50', etc.
    'num_classes': 7,
    'encoder_weights': 'imagenet',

    # Training
    'batch_size': 2, # 8
    'num_epochs': 2, # 150
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,

    # Loss
    'ce_weight': 1.0,
    'dice_weight': 1.0,
    'ignore_index': 6,

    # Hardware
    'num_workers': 4,
    'device': get_device(),

    # Logging
    'checkpoint_dir': str(PROJECT_ROOT / 'outputs' / 'checkpoints'),
    'use_wandb': False,
    'wandb_project': 'deepglobe-segmentation',
    'wandb_run_name': None,  # Auto-generated if None

    # Reproducibility
    'random_seed': 42
}

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Set seed
    set_seed(CONFIG['random_seed'])

    # Initialize wandb
    if CONFIG['use_wandb']:
        run_name = CONFIG['wandb_run_name'] or f"{CONFIG['architecture']}_{CONFIG['encoder']}"
        wandb.init(
            project=CONFIG['wandb_project'],
            name=run_name,
            config=CONFIG
        )

    print("=" * 60)
    print("DeepGlobe Land Cover Segmentation - Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Model: {CONFIG['architecture']} + {CONFIG['encoder']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("=" * 60 + "\n")

    # Load splits
    with open(CONFIG['train_split']) as f:
        train_files = json.load(f)
    with open(CONFIG['val_split']) as f:
        val_files = json.load(f)

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}\n")

    # Create datasets
    train_dataset = DeepGlobeDataset(
        data_dir=CONFIG['data_dir'],
        file_list=train_files,
        tile_size=CONFIG['tile_size'],
        overlap=CONFIG['overlap'],
        transform=DeepGlobeDataset.get_train_transforms(),
        is_train=True
    )

    val_dataset = DeepGlobeDataset(
        data_dir=CONFIG['data_dir'],
        file_list=val_files,
        tile_size=CONFIG['tile_size'],
        overlap=CONFIG['overlap'],
        transform=DeepGlobeDataset.get_val_transforms(),
        is_train=True
    )

    print(f"Training tiles: {len(train_dataset)}")
    print(f"Validation tiles: {len(val_dataset)}\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    # Create model
    model = create_model(
        architecture=CONFIG['architecture'],
        encoder=CONFIG['encoder'],
        num_classes=CONFIG['num_classes'],
        encoder_weights=CONFIG['encoder_weights']
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}\n")

    # Load class weights (computed from EDA)
    # You should compute these from your data and save them
    # For now, using uniform weights
    class_weights = None
    # Example: class_weights = torch.tensor([1.5, 0.8, 2.0, 1.2, 3.0, 4.0, 0.0])

    # Create loss function
    criterion = CombinedLoss(
        class_weights=class_weights,
        ignore_index=CONFIG['ignore_index'],
        ce_weight=CONFIG['ce_weight'],
        dice_weight=CONFIG['dice_weight']
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # First restart after 10 epochs
        T_mult=2,  # Double period after each restart
        eta_min=1e-6
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=CONFIG['device'],
        num_classes=CONFIG['num_classes'],
        ignore_index=CONFIG['ignore_index'],
        checkpoint_dir=CONFIG['checkpoint_dir'],
        use_wandb=CONFIG['use_wandb']
    )

    # Train
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        current_epoch=0,
        early_stopping_patience=CONFIG['early_stopping_patience']
    )

    # Close wandb
    if CONFIG['use_wandb']:
        wandb.finish()

    print("\n☑️ Training completed successfully!")

if __name__ == "__main__":
    main()
