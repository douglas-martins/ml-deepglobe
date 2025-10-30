import numpy as np
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
