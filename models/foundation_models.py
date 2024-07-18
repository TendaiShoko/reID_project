import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(FeatureExtractor, self).__init__()
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.model.fc = nn.Identity()
            self.feature_dim = 2048
        elif model_name == 'vit':
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.model.heads = nn.Identity()
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        return self.model(x)

class ReIDModel(nn.Module):
    def __init__(self, num_classes):
        super(ReIDModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
