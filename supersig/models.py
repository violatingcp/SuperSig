"""Convolutional backbone and supervised CNN shared across all experiments."""
import torch
import torch.nn as nn

from .config import EMB_DIM, N_CLASSES


class ConvBackbone(nn.Module):
    """Shared convolutional feature extractor -> `emb_dim` embedding."""

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SupervisedCNN(nn.Module):
    """Backbone + classification head, trained end-to-end (baseline)."""

    def __init__(self, emb_dim=EMB_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.backbone = ConvBackbone(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class CIFARBackbone(nn.Module):
    """Convolutional feature extractor for 3x32x32 CIFAR images -> `emb_dim` embedding."""

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),      # 32 -> 16
            block(32, 64),     # 16 -> 8
            block(64, 128),    # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CIFARResNetBackbone(nn.Module):
    """
    CIFAR-pretrained ResNet -> `emb_dim` embedding.

    Loads a ResNet trained on CIFAR at 32x32 resolution from torch.hub
    (chenyaofo/pytorch-cifar-models), drops its classification layer, and adds a
    small projection head to `emb_dim`.  The whole network remains trainable, so
    the embedding objective fine-tunes the pretrained features.

    `pretrain` selects the pretraining dataset: "cifar10" (matches the task, but
    note the weights have seen every CIFAR-10 class, including any hold-out) or
    "cifar100" (disjoint label set -- a cleaner initialization for hold-out
    studies).
    """

    def __init__(self, emb_dim=EMB_DIM, arch="resnet20", pretrain="cifar10"):
        super().__init__()
        net = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{pretrain}_{arch}",
                             pretrained=True, trust_repo=True)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.features = net
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))
