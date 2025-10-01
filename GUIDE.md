# DeepGlobe Land Cover Segmentation - Implementation Breakdown

## Project Overview

**Goal:** Automatically segment RGB satellite images (~50cm/pixel) into 7 land cover classes using DeepGlobe 2018 dataset

**Classes:** urban, agriculture, rangeland, forest, water, barren, unknown

**Timeline:** ~3-4 weeks

---

## Phase 1: Project Setup & Environment (Week 1, Days 1-2)

### Step 1: Create Project Structure

```bash
mkdir -p ml-deepglobe-project/{data/{raw,processed,splits},notebooks,src/{data,models,training,utils},configs,outputs/{checkpoints,predictions,figures}}
cd ml-deepglobe-project
```

**Directory structure:**

```
ml-deepglobe-project/
├── data/
│   ├── raw/              # Original DeepGlobe data
│   ├── processed/        # Tiled patches
│   └── splits/           # Train/val split indices
├── notebooks/            # EDA and experiments
├── src/
│   ├── data/            # Data loading & preprocessing
│   ├── models/          # Model definitions
│   ├── training/        # Training loops
│   └── utils/           # Helper functions
├── configs/             # YAML config files
├── outputs/
│   ├── checkpoints/     # Model weights
│   ├── predictions/     # Generated masks
│   └── figures/         # Visualizations
├── requirements.txt
└── README.md
```

### Step 2: Setup Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch albumentations
pip install rasterio opencv-python matplotlib seaborn
pip install pandas numpy scikit-learn tqdm
pip install torchmetrics wandb tensorboard

# Save requirements
pip freeze > requirements.txt
```

### Step 3: Download DeepGlobe Dataset

**Sources:**

- [Kaggle Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
- [Official Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf)

**Expected structure after download:**

```
data/raw/
├── train/
│   ├── *_sat.jpg      # 803 RGB images (2448×2448)
│   └── *_mask.png     # 803 RGB masks
├── valid/             # 171 images (no public masks)
└── test/              # 172 images (no public masks)
```

**Verification script:**

```bash
# Count files
echo "Training images: $(ls data/raw/train/*_sat.jpg | wc -l)"
echo "Training masks: $(ls data/raw/train/*_mask.png | wc -l)"
```

---

## Phase 3: Preprocessing Pipeline (Week 2, Days 1-3)

### Step 6: Create Train/Val Split

**File:** `src/data/create_splits.py`

```python
import numpy as np
import json
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

def get_dominant_class(mask_path, color_map):
    """Get the dominant (most frequent) class in a mask"""
    mask_rgb = np.array(Image.open(mask_path))
    mask_id = rgb_to_class_id(mask_rgb, color_map)

    # Count pixels per class (excluding unknown)
    class_counts = np.bincount(mask_id.flatten(), minlength=7)
    class_counts[6] = 0  # Ignore unknown

    return np.argmax(class_counts)

def create_stratified_split(data_dir, val_ratio=0.15, random_state=42):
    """
    Create stratified train/val split

    Args:
        data_dir: Path to training data directory
        val_ratio: Proportion for validation set
        random_state: Random seed for reproducibility

    Returns:
        train_files, val_files: Lists of file basenames
    """
    COLOR_MAP = {
        (0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
        (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5, (0, 0, 0): 6
    }

    # Get all image files
    mask_paths = sorted(glob.glob(f"{data_dir}/*_mask.png"))
    basenames = [Path(p).stem.replace('_mask', '') for p in mask_paths]

    # Determine dominant class for each image
    print("Analyzing dominant classes for stratification...")
    dominant_classes = []
    for mask_path in tqdm(mask_paths):
        dom_class = get_dominant_class(mask_path, COLOR_MAP)
        dominant_classes.append(dom_class)

    # Stratified split
    train_names, val_names = train_test_split(
        basenames,
        test_size=val_ratio,
        stratify=dominant_classes,
        random_state=random_state
    )

    print(f"\nSplit summary:")
    print(f"  Total: {len(basenames)} images")
    print(f"  Train: {len(train_names)} images ({100*(1-val_ratio):.1f}%)")
    print(f"  Val:   {len(val_names)} images ({100*val_ratio:.1f}%)")

    return train_names, val_names

def save_splits(train_names, val_names, output_dir='data/splits'):
    """Save split indices to JSON files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{output_dir}/train_files.json', 'w') as f:
        json.dump(sorted(train_names), f, indent=2)

    with open(f'{output_dir}/val_files.json', 'w') as f:
        json.dump(sorted(val_names), f, indent=2)

    print(f"\n✓ Splits saved to {output_dir}/")

if __name__ == "__main__":
    train_names, val_names = create_stratified_split('data/raw/train')
    save_splits(train_names, val_names)
```

### Step 7: Implement Tiling Strategy

**File:** `src/data/tiling.py`

```python
import numpy as np
from typing import List, Tuple

def tile_image(image: np.ndarray,
               tile_size: int = 512,
               overlap: int = 64) -> Tuple[List[np.ndarray], List[Tuple]]:
    """
    Split large image into overlapping patches

    Args:
        image: Input image (H, W, C) or (H, W)
        tile_size: Size of each square patch
        overlap: Overlap in pixels between adjacent patches

    Returns:
        patches: List of image patches
        coords: List of (y, x) top-left coordinates for each patch
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    patches = []
    coords = []

    # Calculate number of tiles needed
    n_tiles_y = (h - overlap) // stride
    n_tiles_x = (w - overlap) // stride

    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            y = i * stride
            x = j * stride

            # Extract patch
            patch = image[y:y+tile_size, x:x+tile_size]

            # Only add if patch is exactly tile_size x tile_size
            if patch.shape[0] == tile_size and patch.shape[1] == tile_size:
                patches.append(patch)
                coords.append((y, x))

    return patches, coords

def reconstruct_from_tiles(patches: List[np.ndarray],
                          coords: List[Tuple],
                          original_shape: Tuple,
                          tile_size: int = 512,
                          overlap: int = 64) -> np.ndarray:
    """
    Reconstruct full image from overlapping patches

    Overlapping regions are averaged to smooth transitions

    Args:
        patches: List of predicted patches
        coords: List of (y, x) coordinates
        original_shape: Shape of original image (H, W) or (H, W, C)
        tile_size: Size of each patch
        overlap: Overlap used during tiling

    Returns:
        Reconstructed image
    """
    # Initialize accumulation arrays
    if len(original_shape) == 2:
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
    else:
        h, w, c = original_shape
        reconstructed = np.zeros((h, w, c), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

    # Create blending weights (higher in center, lower at edges)
    blend_mask = create_blend_mask(tile_size, overlap)

    # Accumulate patches
    for patch, (y, x) in zip(patches, coords):
        reconstructed[y:y+tile_size, x:x+tile_size] += patch * blend_mask[:, :, None] if len(original_shape) == 3 else patch * blend_mask
        counts[y:y+tile_size, x:x+tile_size] += blend_mask

    # Average overlapping regions
    counts[counts == 0] = 1  # Avoid division by zero
    if len(original_shape) == 3:
        reconstructed = reconstructed / counts[:, :, None]
    else:
        reconstructed = reconstructed / counts

    return reconstructed.astype(patches[0].dtype)

def create_blend_mask(tile_size: int, overlap: int) -> np.ndarray:
    """
    Create smooth blending mask for overlapping regions

    Center pixels have weight 1.0, edges taper to 0.0 over overlap region
    """
    mask = np.ones((tile_size, tile_size), dtype=np.float32)

    # Create linear ramps for edges
    ramp = np.linspace(0, 1, overlap)

    # Apply ramps to each edge
    mask[:overlap, :] *= ramp[:, None]  # Top
    mask[-overlap:, :] *= ramp[::-1, None]  # Bottom
    mask[:, :overlap] *= ramp[None, :]  # Left
    mask[:, -overlap:] *= ramp[None, ::-1]  # Right

    return mask

# Test tiling
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load test image
    image = np.array(Image.open('data/raw/train/1_sat.jpg'))
    print(f"Original shape: {image.shape}")

    # Tile
    patches, coords = tile_image(image, tile_size=512, overlap=64)
    print(f"Number of patches: {len(patches)}")
    print(f"Patch shape: {patches[0].shape}")

    # Reconstruct
    reconstructed = reconstruct_from_tiles(patches, coords, image.shape)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Calculate reconstruction error
    mse = np.mean((image - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed')
    axes[2].imshow(np.abs(image - reconstructed))
    axes[2].set_title('Absolute Difference')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/figures/tiling_test.png', dpi=150)
    plt.show()
```

### Step 8: Create Dataset Class

**File:** `src/data/dataset.py`

```python
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple

from src.data.mask_conversion import rgb_to_class_id
from src.data.tiling import tile_image

class DeepGlobeDataset(Dataset):
    """
    DeepGlobe Land Cover Dataset with automatic tiling
    """

    COLOR_MAP = {
        (0, 255, 255): 0,    # urban
        (255, 255, 0): 1,    # agriculture
        (255, 0, 255): 2,    # rangeland
        (0, 255, 0): 3,      # forest
        (0, 0, 255): 4,      # water
        (255, 255, 255): 5,  # barren
        (0, 0, 0): 6         # unknown
    }

    def __init__(self,
                 data_dir: str,
                 file_list: List[str],
                 tile_size: int = 512,
                 overlap: int = 64,
                 transform: Optional[A.Compose] = None,
                 is_train: bool = True):
        """
        Args:
            data_dir: Path to data directory containing images and masks
            file_list: List of file basenames (without _sat.jpg or _mask.png)
            tile_size: Size of tiles to extract
            overlap: Overlap between tiles
            transform: Albumentations transforms
            is_train: Whether this is training set (has masks)
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.is_train = is_train

        # Pre-compute all tile coordinates
        self.tiles_info = self._prepare_tiles()

    def _prepare_tiles(self) -> List[Dict]:
        """Cache tile coordinates for all images"""
        tiles_info = []

        for file_id in self.file_list:
            # Load image to get shape
            img_path = self.data_dir / f"{file_id}_sat.jpg"
            image = np.array(Image.open(img_path))
            h, w = image.shape[:2]

            # Calculate tile coordinates
            stride = self.tile_size - self.overlap
            n_tiles_y = (h - self.overlap) // stride
            n_tiles_x = (w - self.overlap) // stride

            for i in range(n_tiles_y):
                for j in range(n_tiles_x):
                    y = i * stride
                    x = j * stride

                    tiles_info.append({
                        'file_id': file_id,
                        'y': y,
                        'x': x
                    })

        return tiles_info

    def __len__(self) -> int:
        return len(self.tiles_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (3, tile_size, tile_size)
            mask: Tensor of shape (tile_size, tile_size) with class IDs
        """
        tile_info = self.tiles_info[idx]
        file_id = tile_info['file_id']
        y = tile_info['y']
        x = tile_info['x']

        # Load full image
        img_path = self.data_dir / f"{file_id}_sat.jpg"
        image = np.array(Image.open(img_path))

        # Extract tile
        tile_img = image[y:y+self.tile_size, x:x+self.tile_size]

        if self.is_train:
            # Load and convert mask
            mask_path = self.data_dir / f"{file_id}_mask.png"
            mask_rgb = np.array(Image.open(mask_path))
            mask_id = rgb_to_class_id(mask_rgb, self.COLOR_MAP)
            tile_mask = mask_id[y:y+self.tile_size, x:x+self.tile_size]
        else:
            # No mask for inference
            tile_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=tile_img, mask=tile_mask)
            tile_img = transformed['image']
            tile_mask = transformed['mask']

        return tile_img, tile_mask.long()

    @staticmethod
    def get_train_transforms() -> A.Compose:
        """Training augmentations"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                         saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transforms() -> A.Compose:
        """Validation transforms (no augmentation)"""
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# Example usage
if __name__ == "__main__":
    # Load split files
    with open('data/splits/train_files.json') as f:
        train_files = json.load(f)

    # Create dataset
    train_dataset = DeepGlobeDataset(
        data_dir='data/raw/train',
        file_list=train_files[:10],  # Test with 10 files
        transform=DeepGlobeDataset.get_train_transforms()
    )

    print(f"Dataset size: {len(train_dataset)} tiles")

    # Test loading
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}")  # [3, 512, 512]
    print(f"Mask shape: {mask.shape}")    # [512, 512]
    print(f"Unique classes: {torch.unique(mask)}")
```

### Step 9: Test DataLoader Pipeline

**File:** `notebooks/02_test_dataloader.ipynb`

```python
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

from src.data.dataset import DeepGlobeDataset

# Load splits
with open('data/splits/train_files.json') as f:
    train_files = json.load(f)
with open('data/splits/val_files.json') as f:
    val_files = json.load(f)

# Create datasets
train_dataset = DeepGlobeDataset(
    data_dir='data/raw/train',
    file_list=train_files,
    transform=DeepGlobeDataset.get_train_transforms(),
    is_train=True
)

val_dataset = DeepGlobeDataset(
    data_dir='data/raw/train',
    file_list=val_files,
    transform=DeepGlobeDataset.get_val_transforms(),
    is_train=True
)

print(f"Train tiles: {len(train_dataset)}")
print(f"Val tiles: {len(val_dataset)}")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Test iteration
batch = next(iter(train_loader))
images, masks = batch

print(f"\nBatch shapes:")
print(f"  Images: {images.shape}")  # [8, 3, 512, 512]
print(f"  Masks: {masks.shape}")    # [8, 512, 512]
print(f"  Image dtype: {images.dtype}")
print(f"  Mask dtype: {masks.dtype}")
print(f"  Unique classes in batch: {torch.unique(masks)}")

# Denormalize for visualization
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

# Visualize batch
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i in range(4):
    # Denormalize image
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    # Get mask
    mask = masks[i].numpy()

    # Plot
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f'Image {i+1}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(mask, cmap='tab10', vmin=0, vmax=6)
    axes[i, 1].set_title(f'Mask {i+1}')
    axes[i, 1].axis('off')

    # Show class distribution
    unique, counts = np.unique(mask, return_counts=True)
    axes[i, 2].bar(unique, counts)
    axes[i, 2].set_title(f'Class Distribution {i+1}')
    axes[i, 2].set_xlabel('Class ID')
    axes[i, 2].set_ylabel('Pixel Count')

    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('outputs/figures/dataloader_batch_test.png', dpi=150)
plt.show()

print("\n✓ DataLoader pipeline working correctly!")
```

---

## Phase 4: Initial Model Setup (Week 2, Days 4-5)

### Step 10: Create Model Wrapper

**File:** `src/models/segmentation_model.py`

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Literal

def create_model(
    architecture: Literal['unet', 'deeplabv3plus'] = 'unet',
    encoder: str = 'resnet34',
    num_classes: int = 7,
    encoder_weights: str = 'imagenet',
    activation: str = None
) -> nn.Module:
    """
    Create segmentation model using SMP

    Args:
        architecture: Model architecture ('unet', 'deeplabv3plus')
        encoder: Encoder backbone ('resnet34', 'resnet50', 'efficientnet-b3', etc.)
        num_classes: Number of output classes
        encoder_weights: Pretrained weights ('imagenet', None)
        activation: Output activation (None for logits, 'softmax', 'sigmoid')

    Returns:
        PyTorch model
    """
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test model creation
if __name__ == "__main__":
    print("Testing model architectures...\n")

    models_to_test = [
        ('unet', 'resnet34'),
        ('unet', 'resnet50'),
        ('deeplabv3plus', 'resnet34'),
        ('deeplabv3plus', 'resnet50'),
    ]

    for arch, encoder in models_to_test:
        model = create_model(
            architecture=arch,
            encoder=encoder,
            num_classes=7,
            encoder_weights='imagenet'
        )

        # Test forward pass
        dummy_input = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)

        num_params = count_parameters(model)

        print(f"{arch:15s} + {encoder:12s}:")
        print(f"  Input shape:  {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Parameters:   {num_params:,}")
        print()

    print("✓ All models working correctly!")
```

### Step 11: Setup Loss Functions

**File:** `src/training/losses.py`

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

class CombinedLoss(nn.Module):
    """
    Combination of Cross-Entropy and Dice Loss

    Useful for handling class imbalance in segmentation
    """
    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 6,
                 ce_weight: float = 1.0,
                 dice_weight: float = 1.0):
        """
        Args:
            class_weights: Weights for each class in CE loss
            ignore_index: Class to ignore (unknown = 6)
            ce_weight: Weight for CE loss component
            dice_weight: Weight for Dice loss component
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            ignore_index=ignore_index
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predictions (B, C, H, W) - logits
            targets: Ground truth (B, H, W) - class IDs

        Returns:
            Combined loss value
        """
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)

        return self.ce_weight * ce + self.dice_weight * dice

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Focuses on hard examples by down-weighting easy ones
    """
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 ignore_index: int = 6):
        """
        Args:
            alpha: Weights for each class
            gamma: Focusing parameter (0 = CE loss, >0 = more focus on hard examples)
            ignore_index: Class to ignore
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predictions (B, C, H, W) - logits
            targets: Ground truth (B, H, W) - class IDs
        """
        ce_loss = nn.functional.cross_entropy(
            preds, targets,
            reduction='none',
            ignore_index=self.ignore_index
        )

        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

def compute_class_weights(class_counts: torch.Tensor,
                         ignore_index: int = 6,
                         smoothing: float = 1.0) -> torch.Tensor:
    """
    Compute inverse frequency weights for loss function

    Args:
        class_counts: Tensor of pixel counts per class
        ignore_index: Class to ignore (set weight to 0)
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        Tensor of class weights
    """
    # Create copy to avoid modifying original
    counts = class_counts.clone().float()

    # Set ignored class count to 0
    counts[ignore_index] = 0

    # Inverse frequency with smoothing
    weights = 1.0 / (counts + smoothing)

    # Normalize so sum equals number of active classes
    active_classes = (counts > 0).sum().item()
    weights = weights / weights.sum() * active_classes

    # Zero out ignored class
    weights[ignore_index] = 0

    return weights

# Test losses
if __name__ == "__main__":
    # Mock data
    batch_size = 4
    num_classes = 7
    h, w = 512, 512

    preds = torch.randn(batch_size, num_classes, h, w)
    targets = torch.randint(0, num_classes, (batch_size, h, w))

    # Test class weights computation
    class_counts = torch.tensor([1000000, 5000000, 800000, 2000000, 500000, 300000, 100000])
    weights = compute_class_weights(class_counts)
    print("Class weights:")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.4f}")
    print()

    # Test losses
    print("Testing loss functions...\n")

    # Combined Loss
    combined_loss = CombinedLoss(class_weights=weights)
    loss_val = combined_loss(preds, targets)
    print(f"Combined Loss: {loss_val.item():.4f}")

    # Focal Loss
    focal_loss = FocalLoss(alpha=weights, gamma=2.0)
    loss_val = focal_loss(preds, targets)
    print(f"Focal Loss: {loss_val.item():.4f}")

    print("\n✓ Loss functions working correctly!")
```

### Step 12: Setup Metrics

**File:** `src/training/metrics.py`

```python
import torch
import numpy as np
from typing import Dict, List
from torchmetrics import JaccardIndex, Accuracy

class SegmentationMetrics:
    """
    Compute segmentation metrics following DeepGlobe protocol

    - mIoU (mean Intersection over Union) excluding 'unknown' class
    - Per-class IoU
    - Overall Accuracy
    """
    def __init__(self, num_classes: int = 7, ignore_index: int = 6, device: str = 'cuda'):
        """
        Args:
            num_classes: Total number of classes (including unknown)
            ignore_index: Class to ignore in metrics (unknown = 6)
            device: Device for computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        # Initialize metrics
        self.iou_metric = JaccardIndex(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='none'  # Per-class IoU
        ).to(device)

        self.accuracy_metric = Accuracy(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='micro'
        ).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions

        Args:
            preds: Predictions (B, C, H, W) - logits or (B, H, W) - class IDs
            targets: Ground truth (B, H, W) - class IDs
        """
        # Convert logits to class predictions if needed
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        # Update metrics
        self.iou_metric.update(preds, targets)
        self.accuracy_metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics

        Returns:
            Dictionary with mIoU, per-class IoU, and accuracy
        """
        # Compute per-class IoU
        per_class_iou = self.iou_metric.compute()

        # Compute mIoU (excluding unknown class)
        # Filter out NaN values (classes not present)
        valid_ious = per_class_iou[~torch.isnan(per_class_iou)]
        miou = valid_ious.mean().item()

        # Compute accuracy
        accuracy = self.accuracy_metric.compute().item()

        # Build results dictionary
        results = {
            'mIoU': miou,
            'accuracy': accuracy
        }

        # Add per-class IoU
        class_names = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren', 'unknown']
        for i, name in enumerate(class_names):
            if i != self.ignore_index:
                iou_val = per_class_iou[i].item()
                results[f'IoU_{name}'] = iou_val

        return results

    def reset(self):
        """Reset metrics for new epoch"""
        self.iou_metric.reset()
        self.accuracy_metric.reset()

# Test metrics
if __name__ == "__main__":
    print("Testing segmentation metrics...\n")

    # Mock data
    batch_size = 4
    num_classes = 7
    h, w = 512, 512

    # Create perfect predictions for testing
    targets = torch.randint(0, num_classes, (batch_size, h, w))
    preds = torch.nn.functional.one_hot(targets, num_classes=num_classes)
    preds = preds.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Add some noise
    noise = torch.randn_like(preds) * 0.1
    preds = preds + noise

    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=7, device='cpu')

    # Update metrics
    metrics.update(preds, targets)

    # Compute
    results = metrics.compute()

    print("Metrics Results:")
    print(f"  mIoU: {results['mIoU']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print("\nPer-class IoU:")
    for key, value in results.items():
        if key.startswith('IoU_'):
            print(f"  {key}: {value:.4f}")

    print("\n✓ Metrics working correctly!")
```

---

## Phase 5: Training Setup (Week 3, Days 1-2)

### Step 13: Create Training Loop

**File:** `src/training/trainer.py`

```python
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
            if (epoch + 1) % 10 == 0:
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
```

### Step 14: Create Training Script

**File:** `train.py`

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import wandb
from pathlib import Path

from src.data.dataset import DeepGlobeDataset
from src.models.segmentation_model import create_model
from src.training.losses import CombinedLoss, compute_class_weights
from src.training.trainer import Trainer

# Configuration
CONFIG = {
    # Data
    'data_dir': 'data/raw/train',
    'train_split': 'data/splits/train_files.json',
    'val_split': 'data/splits/val_files.json',
    'tile_size': 512,
    'overlap': 64,

    # Model
    'architecture': 'unet',  # 'unet' or 'deeplabv3plus'
    'encoder': 'resnet34',   # 'resnet34', 'resnet50', etc.
    'num_classes': 7,
    'encoder_weights': 'imagenet',

    # Training
    'batch_size': 8,
    'num_epochs': 150,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,

    # Loss
    'ce_weight': 1.0,
    'dice_weight': 1.0,
    'ignore_index': 6,

    # Hardware
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Logging
    'checkpoint_dir': 'outputs/checkpoints',
    'use_wandb': True,
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
        early_stopping_patience=CONFIG['early_stopping_patience']
    )

    # Close wandb
    if CONFIG['use_wandb']:
        wandb.finish()

    print("\n✓ Training completed successfully!")

if __name__ == "__main__":
    main()
```

**Run training:**

```bash
python train.py
```

---

## Phase 6: Evaluation & Visualization (Week 3, Days 3-5)

### Step 15: Create Inference Script

**File:** `inference.py`

```python
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

from src.models.segmentation_model import create_model
from src.data.dataset import DeepGlobeDataset
from src.data.tiling import tile_image, reconstruct_from_tiles
from src.data.mask_conversion import class_id_to_rgb

@torch.no_grad()
def predict_image(model, image_path, device='cuda', tile_size=512, overlap=64):
    """
    Predict segmentation mask for full image

    Args:
        model: Trained segmentation model
        image_path: Path to input image
        device: Device for inference
        tile_size: Size of tiles
        overlap: Overlap between tiles

    Returns:
        Predicted mask (H, W) with class IDs
    """
    model.eval()

    # Load image
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    # Tile image
    patches, coords = tile_image(image, tile_size, overlap)

    # Prepare transforms (same as validation)
    transforms = DeepGlobeDataset.get_val_transforms()

    # Predict each patch
    predicted_patches = []
    for patch in tqdm(patches, desc="Predicting tiles", leave=False):
        # Transform
        transformed = transforms(image=patch)
        patch_tensor = transformed['image'].unsqueeze(0).to(device)

        # Predict
        output = model(patch_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        predicted_patches.append(pred_mask)

    # Reconstruct full mask
    full_mask = reconstruct_from_tiles(
        predicted_patches,
        coords,
        (h, w),
        tile_size,
        overlap
    )

    return full_mask.astype(np.uint8)

def main():
    """Run inference on test set"""
    # Configuration
    CONFIG = {
        'checkpoint_path': 'outputs/checkpoints/best_model.pth',
        'architecture': 'unet',
        'encoder': 'resnet34',
        'num_classes': 7,
        'test_dir': 'data/raw/valid',  # or 'data/raw/test'
        'output_dir': 'outputs/predictions',
        'tile_size': 512,
        'overlap': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("DeepGlobe - Inference")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Checkpoint: {CONFIG['checkpoint_path']}")
    print("=" * 60 + "\n")

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = create_model(
        architecture=CONFIG['architecture'],
        encoder=CONFIG['encoder'],
        num_classes=CONFIG['num_classes'],
        encoder_weights=None  # Not needed for inference
    )

    # Load checkpoint
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['device'])

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Best mIoU: {checkpoint['best_miou']:.4f}\n")

    # Get test images
    test_dir = Path(CONFIG['test_dir'])
    test_images = sorted(test_dir.glob('*_sat.jpg'))

    print(f"Found {len(test_images)} images to process\n")

    # Color map for visualization
    COLOR_MAP = {
        (0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
        (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5, (0, 0, 0): 6
    }

    # Process each image
    for img_path in tqdm(test_images, desc="Processing images"):
        # Predict
        pred_mask = predict_image(
            model,
            img_path,
            CONFIG['device'],
            CONFIG['tile_size'],
            CONFIG['overlap']
        )

        # Convert to RGB for visualization
        pred_mask_rgb = class_id_to_rgb(pred_mask, COLOR_MAP)

        # Save
        basename = img_path.stem.replace('_sat', '')
        output_path = output_dir / f'{basename}_pred.png'
        Image.fromarray(pred_mask_rgb).save(output_path)

    print(f"\n✓ Predictions saved to {output_dir}/")

if __name__ == "__main__":
    main()
```

### Step 16: Create Visualization Utilities

**File:** `src/utils/visualization.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from typing import List, Tuple

# Define class colors and names
CLASS_COLORS = np.array([
    [0, 255, 255],    # 0: urban (cyan)
    [255, 255, 0],    # 1: agriculture (yellow)
    [255, 0, 255],    # 2: rangeland (magenta)
    [0, 255, 0],      # 3: forest (green)
    [0, 0, 255],      # 4: water (blue)
    [255, 255, 255],  # 5: barren (white)
    [0, 0, 0]         # 6: unknown (black)
]) / 255.0

CLASS_NAMES = [
    'Urban', 'Agriculture', 'Rangeland',
    'Forest', 'Water', 'Barren', 'Unknown'
]

def visualize_prediction(image: np.ndarray,
                        true_mask: np.ndarray,
                        pred_mask: np.ndarray,
                        save_path: str = None,
                        title: str = "Segmentation Results"):
    """
    Visualize image, ground truth, and prediction side by side

    Args:
        image: RGB image (H, W, 3)
        true_mask: Ground truth mask (H, W) with class IDs
        pred_mask: Predicted mask (H, W) with class IDs
        save_path: Path to save figure
        title: Figure title
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Ground truth
    cmap = ListedColormap(CLASS_COLORS)
    axes[1].imshow(true_mask, cmap=cmap, vmin=0, vmax=6)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=6)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Difference (errors)
    diff = (true_mask != pred_mask).astype(float)
    diff[true_mask == 6] = np.nan  # Ignore unknown class
    axes[3].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title('Errors (Red)')
    axes[3].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def create_legend(save_path: str = None):
    """Create a legend for class colors"""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create dummy data for legend
    for i, (color, name) in enumerate(zip(CLASS_COLORS[:-1], CLASS_NAMES[:-1])):
        ax.scatter([], [], c=[color], s=200, label=name)

    ax.legend(loc='center', fontsize=12, frameon=True, ncol=2)
    ax.axis('off')

    plt.title('Land Cover Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def plot_training_curves(history_path: str, save_path: str = None):
    """
    Plot training and validation curves from history

    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure
    """
    import json

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(len(history['train_loss']))

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # mIoU
    axes[1].plot(epochs, history['train_miou'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_miou'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Mean Intersection over Union')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history['lr'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def create_error_atlas(image_paths: List[str],
                       true_mask_paths: List[str],
                       pred_mask_paths: List[str],
                       save_path: str,
                       num_samples: int = 10):
    """
    Create an atlas showing predictions with highest errors

    Args:
        image_paths: List of image paths
        true_mask_paths: List of ground truth mask paths
        pred_mask_paths: List of predicted mask paths
        save_path: Path to save atlas
        num_samples: Number of samples to show
    """
    # Calculate errors for each image
    errors = []
    for true_path, pred_path in zip(true_mask_paths, pred_mask_paths):
        true_mask = np.array(Image.open(true_path))
        pred_mask = np.array(Image.open(pred_path))

        # Calculate error rate (excluding unknown)
        valid = true_mask != 6
        if valid.sum() > 0:
            error_rate = (true_mask[valid] != pred_mask[valid]).mean()
        else:
            error_rate = 0.0

        errors.append(error_rate)

    # Get indices of highest errors
    error_indices = np.argsort(errors)[-num_samples:][::-1]

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i, idx in enumerate(error_indices):
        image = np.array(Image.open(image_paths[idx]))
        true_mask = np.array(Image.open(true_mask_paths[idx]))
        pred_mask = np.array(Image.open(pred_mask_paths[idx]))

        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Sample {idx} - Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_mask)
        axes[i, 1].set_title(f'Ground Truth (Error: {errors[idx]:.2%})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### Step 17: Create Evaluation Notebook

**File:** `notebooks/03_evaluate_results.ipynb`

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image

from src.utils.visualization import (
    visualize_prediction, create_legend,
    plot_training_curves, create_error_atlas
)
from src.training.metrics import SegmentationMetrics

# ===== 1. Plot Training Curves =====
print("Plotting training curves...")
plot_training_curves(
    'outputs/checkpoints/training_history.json',
    save_path='outputs/figures/training_curves.png'
)

# ===== 2. Create Class Legend =====
print("Creating class legend...")
create_legend(save_path='outputs/figures/class_legend.png')

# ===== 3. Evaluate on Validation Set =====
print("\nEvaluating on validation set...")

import torch
from torch.utils.data import DataLoader
from src.data.dataset import DeepGlobeDataset
from src.models.segmentation_model import create_model

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(architecture='unet', encoder='resnet34', num_classes=7)
checkpoint = torch.load('outputs/checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load validation data
with open('data/splits/val_files.json') as f:
    val_files = json.load(f)

val_dataset = DeepGlobeDataset(
    data_dir='data/raw/train',
    file_list=val_files,
    transform=DeepGlobeDataset.get_val_transforms(),
    is_train=True
)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Compute metrics
metrics_tracker = SegmentationMetrics(num_classes=7, device=device)

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        metrics_tracker.update(outputs, masks)

final_metrics = metrics_tracker.compute()

print("\n" + "=" * 60)
print("FINAL VALIDATION METRICS")
print("=" * 60)
print(f"mIoU: {final_metrics['mIoU']:.4f}")
print(f"Accuracy: {final_metrics['accuracy']:.4f}")
print("\nPer-class IoU:")
for key, value in final_metrics.items():
    if key.startswith('IoU_'):
        class_name = key.replace('IoU_', '')
        print(f"  {class_name:12s}: {value:.4f}")
print("=" * 60)

# ===== 4. Visualize Sample Predictions =====
print("\nVisualizing sample predictions...")

# Get a few validation samples
sample_indices = [0, 100, 200, 300, 400]

for idx in sample_indices:
    image_tensor, mask = val_dataset[idx]

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image_tensor * std + mean).permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    # Predict
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Visualize
    visualize_prediction(
        image,
        mask.numpy(),
        pred,
        save_path=f'outputs/figures/prediction_sample_{idx}.png',
        title=f'Validation Sample {idx}'
    )

print("\n✓ Evaluation completed!")
print("Check outputs/figures/ for visualizations")
```

---

## Summary: Complete Workflow Checklist

### Week 1: Setup & Data Exploration

- [x] Setup project structure and environment
- [x] Download DeepGlobe dataset
- [x] Complete EDA notebook:
  - [x] Analyze images and masks
  - [x] Define color mappings
  - [x] Calculate class distribution
  - [x] Compute class weights
- [x] Test RGB-to-ID conversion
- [x] Document findings

### Week 2: Data Pipeline & Model Setup

- [ ] Create stratified train/val split
- [ ] Implement tiling/reconstruction functions
- [ ] Build Dataset class with augmentations
- [ ] Test DataLoader pipeline
- [ ] Create model wrapper (U-Net, DeepLabV3+)
- [ ] Setup loss functions (Combined, Focal)
- [ ] Setup metrics tracker
- [ ] Test forward/backward passes

### Week 3: Training & Evaluation

- [ ] Implement training loop
- [ ] Create training script
- [ ] Train baseline model (U-Net ResNet34)
- [ ] Monitor with W&B/TensorBoard
- [ ] Implement inference script
- [ ] Create visualization utilities
- [ ] Evaluate on validation set
- [ ] Generate error analysis

### Week 4: Experimentation & Reporting

- [ ] Ablation studies:
  - [ ] Different architectures
  - [ ] Different loss combinations
  - [ ] Augmentation importance
  - [ ] Effect of class weights
- [ ] Generate predictions on test set
- [ ] Create final visualizations
- [ ] Write report with:
  - [ ] Problem description
  - [ ] Methodology
  - [ ] Results and metrics
  - [ ] Discussion and limitations
  - [ ] Future work

---

## Key Files Summary

| File                               | Purpose                     |
| ---------------------------------- | --------------------------- |
| `src/data/mask_conversion.py`      | RGB ↔ ID conversion         |
| `src/data/tiling.py`               | Image tiling/reconstruction |
| `src/data/create_splits.py`        | Train/val split creation    |
| `src/data/dataset.py`              | PyTorch Dataset class       |
| `src/models/segmentation_model.py` | Model creation              |
| `src/training/losses.py`           | Loss functions              |
| `src/training/metrics.py`          | Evaluation metrics          |
| `src/training/trainer.py`          | Training manager            |
| `src/utils/visualization.py`       | Plotting utilities          |
| `train.py`                         | Main training script        |
| `inference.py`                     | Prediction script           |

---

## Best Practices

1. **Start Small**: Test with 10-20 images before full training
2. **Monitor Overfitting**: Watch train vs val metrics closely
3. **Save Checkpoints**: Save every 10 epochs and best model
4. **Use Mixed Precision**: Add `torch.cuda.amp` for faster training
5. **Experiment Tracking**: Always use W&B or TensorBoard
6. **Ablation Studies**: Change one thing at a time
7. **Visual Inspection**: Always look at predictions, not just metrics
8. **Document Everything**: Keep notes on what works and what doesn't

---

## Resources

- [DeepGlobe Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Demir_DeepGlobe_2018_A_CVPR_2018_paper.pdf)
- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [Albumentations Docs](https://albumentations.ai/docs/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)
- [Weights & Biases](https://wandb.ai/)
