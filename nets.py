import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet18", "resnet50", "efficientnet_b0")


def build_model(name, num_classes, pretrained=True):
    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model không hỗ trợ: {name}. Chọn một trong {SUPPORTED}")
    return m


def head(model, name):
    return model.fc if name.startswith("resnet") else model.classifier


def target_layer(model, name):
    """Lớp tích chập cuối, dùng cho Grad-CAM."""
    return model.layer4[-1] if name.startswith("resnet") else model.features[-1]


def set_backbone_trainable(model, name, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head(model, name).parameters():
        p.requires_grad = True
