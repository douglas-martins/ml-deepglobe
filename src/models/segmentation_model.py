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
