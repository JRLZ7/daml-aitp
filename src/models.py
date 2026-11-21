"""Explicitly sourced ImageNet encoder and standalone checkpoint loading."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from .preprocessing import preprocessing_from_checkpoint

WEIGHT_NAME = 'ResNet50_Weights.IMAGENET1K_V2'


def baseline_weights(name):
    if name != WEIGHT_NAME:
        raise ValueError(f'Baseline requires explicit {WEIGHT_NAME}')
    return ResNet50_Weights.IMAGENET1K_V2


class SceneClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.encoder = resnet50(weights=baseline_weights(WEIGHT_NAME) if pretrained else None)
        dimensions = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(dimensions, num_classes)

    def forward(self, images):
        return self.head(self.encoder(images))


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    architecture = checkpoint['architecture']
    if architecture == 'resnet18':
        classes = checkpoint['class_names']
        if len(classes) != len(set(classes)):
            raise ValueError('Checkpoint has duplicate class labels')
        if checkpoint['class_to_idx'] != {label: i for i, label in enumerate(classes)}:
            raise ValueError('Checkpoint class mapping differs from class order')
        if checkpoint['weights'] != 'IMAGENET1K_V1':
            raise ValueError('Historical ResNet18 requires ImageNet IMAGENET1K_V1 weights')
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        if (checkpoint['image_size'] != 224 or
                list(checkpoint['resize_size']) != list(transform.resize_size) or
                list(checkpoint['mean']) != list(transform.mean) or
                list(checkpoint['std']) != list(transform.std)):
            raise ValueError('Checkpoint preprocessing differs from ImageNet ResNet18 transforms')
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        # Normalize the historical schema for the shared prediction interface.
        checkpoint = {**checkpoint, 'classes': classes,
                      'pretrained_weights': 'ResNet18_Weights.IMAGENET1K_V1'}
        model.to(device).eval()
        return model, transform, checkpoint
    if architecture != 'resnet50':
        raise ValueError('Unsupported checkpoint architecture')
    classes = checkpoint['classes']
    if len(classes) != len(set(classes)):
        raise ValueError('Checkpoint has duplicate class labels')
    model = SceneClassifier(len(classes), pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    transform = preprocessing_from_checkpoint(checkpoint, baseline_weights(checkpoint['pretrained_weights']))
    return model, transform, checkpoint
