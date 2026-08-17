"""
model.py — rebuilds the EfficientNet-B2 architecture used during training.

A .pth file only stores the learned weights, not the model's structure. This
function recreates that exact structure, so the saved weights have a matching
"shape" to be loaded into. It is the same architecture function used when the
model was originally trained and saved.
"""
import torch
import torchvision
from torch import nn


def create_effnetb2_model(num_classes=3, seed=42):
    # Load EfficientNet-B2 along with its pretrained ImageNet weights
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model_transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze the pretrained backbone (this is what makes it a feature extractor)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head to match our number of classes
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return model, model_transforms
