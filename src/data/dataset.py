import sys, os
from collections import OrderedDict
# from functools import lru_cache

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple

from src.data.mask_conversion import rgb_to_class_id, rgb_tile_to_id

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

        # Cache optimized tiles
        # self._get_image = lru_cache(maxsize=32)(self._load_image)
        # self._get_mask = lru_cache(maxsize=32)(self._load_mask) if is_train else None
        self.cache_size = 32  # adjust if you’d like
        self._image_cache = OrderedDict()
        self._mask_cache = OrderedDict() if is_train else None


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

    def _load_image(self, file_id: str) -> np.ndarray:
        img_path = self.data_dir / f"{file_id}_sat.jpg"
        return np.array(Image.open(img_path))

    def _load_mask(self, file_id: str) -> np.ndarray:
        mask_path = self.data_dir / f"{file_id}_mask.png"
        return np.array(Image.open(mask_path).convert("RGB"))

    def _get_image(self, file_id: str) -> np.ndarray:
        cache = self._image_cache
        if file_id in cache:
            cache.move_to_end(file_id)
            return cache[file_id]
        image = self._load_image(file_id)
        cache[file_id] = image
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return image

    def _get_mask(self, file_id: str) -> np.ndarray:
        cache = self._mask_cache
        if cache is None:
            raise RuntimeError("Mask cache requested in inference mode")
        if file_id in cache:
            cache.move_to_end(file_id)
            return cache[file_id]
        mask = self._load_mask(file_id)
        cache[file_id] = mask
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return mask

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
        # img_path = self.data_dir / f"{file_id}_sat.jpg"
        # image = np.array(Image.open(img_path))
        image = self._get_image(file_id)

        # Extract tile
        tile_img = image[y:y+self.tile_size, x:x+self.tile_size]

        if self.is_train:
            # Load and convert mask
            # mask_path = self.data_dir / f"{file_id}_mask.png"
            # mask_rgb = np.array(Image.open(mask_path))
            # mask_id = rgb_to_class_id(mask_rgb, self.COLOR_MAP)
            mask_rgb = self._get_mask(file_id)
            mask_tile_rgb = mask_rgb[y:y+self.tile_size, x:x+self.tile_size]
            tile_mask = rgb_tile_to_id(mask_tile_rgb, self.COLOR_MAP)
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
