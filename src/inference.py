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
from src.utils.torch_device import get_device

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
        'checkpoint_path': '../outputs/checkpoints/best_model.pth',
        'architecture': 'unet',
        'encoder': 'resnet34',
        'num_classes': 7,
        'test_dir': '../data/raw/valid',  # or 'data/raw/test'
        'output_dir': '../outputs/predictions',
        'tile_size': 512,
        'overlap': 64,
        'device': get_device()
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

    print(f"☑️ Model loaded from epoch {checkpoint['epoch']}")
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

    print(f"\n☑️ Predictions saved to {output_dir}/")

if __name__ == "__main__":
    main()
