import numpy as np
import json
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

from typing import Dict, Tuple


def rgb_to_class_id(mask_rgb: np.ndarray,
                    color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """
    Convert RGB mask to class ID mask

    Args:
        mask_rgb: RGB mask array (H, W, 3)
        color_map: Dictionary mapping RGB tuples to class IDs

    Returns:
        Class ID mask array (H, W)
    """
    h, w = mask_rgb.shape[:2]
    mask_id = np.zeros((h, w), dtype=np.uint8)

    for color, class_id in color_map.items():
        match = np.all(mask_rgb == color, axis=-1)
        mask_id[match] = class_id

    return mask_id


def get_dominant_class(mask_path, color_map):
    """Get the dominant (most frequent) class in a mask"""
    mask_rgb = np.array(Image.open(mask_path))
    mask_id = rgb_to_class_id(mask_rgb, color_map)

    # Count pixels per class (excluding unknown)
    class_counts = np.bincount(mask_id.flatten(), minlength=7)
    class_counts[6] = 0  # Ignore unknown

    return int(np.argmax(class_counts))


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


def save_splits(train_names, val_names, output_dir='../data/splits'):
    """Save split indices to JSON files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f'{output_dir}/train_files.json', 'w') as f:
        json.dump(sorted(train_names), f, indent=2)

    with open(f'{output_dir}/val_files.json', 'w') as f:
        json.dump(sorted(val_names), f, indent=2)

    print(f"\n✓ Splits saved to {output_dir}/")
