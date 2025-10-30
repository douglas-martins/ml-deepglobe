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
