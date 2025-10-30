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
