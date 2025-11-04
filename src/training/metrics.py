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
        ).to('cpu')

        self.accuracy_metric = Accuracy(
            task='multiclass',
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='micro'
        ).to('cpu')

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
