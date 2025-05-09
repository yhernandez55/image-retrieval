import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetTransferModel(nn.Module):
    def __init__(self, num_classes=101, embedding_size=512, pretrained=True):
        super().__init__()

        # Update to use the `weights` argument
        self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        # Freeze ONLY first 2 blocks
        for name, param in self.backbone.named_parameters():
            if not name.startswith(('layer1', 'layer2')):
                param.requires_grad = False

        self.backbone.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(embedding_size, embedding_size),  # Additional layer
            nn.LayerNorm(embedding_size)  # Better than BN for embeddings
        )
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # Extract features from ResNet
        features = self.backbone(x)
        # Get embedding
        embedding = self.embedding(features)
        # Get class predictions
        logits = self.classifier(embedding), F.normalize(embedding, p=2, dim=1)
        return logits  # Return logits + L2-normalized embeddings

    def extract_features(self, x):
        """Extract feature embeddings for image retrieval"""
        features = self.backbone(x)
        embedding = self.embedding(features)
        # Normalize embedding to unit length for better similarity search
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding
