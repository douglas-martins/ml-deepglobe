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

# def rgb_tile_to_id(tile_rgb: np.ndarray,
#                    color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
#     """
#     Convert RGB tile to class ID tile

#     Args:
#         tile_rgb: RGB tile array (H, W, 3)
#         color_map: Dictionary mapping RGB tuples to class IDs

#     Returns:
#         Class ID tile array (H, W)
#     """
#     view = tile_rgb.view([('r', tile_rgb.dtype),
#                           ('g', tile_rgb.dtype),
#                           ('b', tile_rgb.dtype)])
#     id_tile = np.zeros(tile_rgb.shape[:2], dtype=np.uint8)
#     for color, class_id in color_map.items():
#         packed = np.array(color, dtype=tile_rgb.dtype).view(view.dtype)
#         id_tile[view == packed] = class_id
#     return id_tile

def rgb_tile_to_id(tile_rgb: np.ndarray,
                   color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """
    Convert RGB tile to class ID tile

    Args:
        tile_rgb: RGB tile array (H, W, 3)
        color_map: Dictionary mapping RGB tuples to class IDs

    Returns:
        Class ID tile array (H, W)
    """
    if tile_rgb.ndim != 3 or tile_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB tile with shape (H, W, 3), got {tile_rgb.shape}")

    id_tile = np.zeros(tile_rgb.shape[:2], dtype=np.uint8)

    for color, class_id in color_map.items():
        match = np.all(tile_rgb == color, axis=-1)
        id_tile[match] = class_id

    return id_tile
