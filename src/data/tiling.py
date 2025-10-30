import numpy as np
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt

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
