# src/models/fusion_model.py
import torch
import torch.nn as nn
from .vision_transformer import VisionTransformer
from .tabular_mlp import TabularMLP

class SpatiotemporalFusionModel(nn.Module):
    def __init__(self, vit_model, tabular_model, vit_output_dim, tabular_output_dim, num_classes=1):
        super().__init__()
        self.vit = vit_model
        self.tabular_model = tabular_model
        self.fusion_head = nn.Sequential(
            nn.Linear(vit_output_dim + tabular_output_dim, 512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, image_data, tabular_data):
        vit_features = self.vit(image_data)
        tabular_features = self.tabular_model(tabular_data)
        fused_features = torch.cat((vit_features, tabular_features), dim=1)
        output = self.fusion_head(fused_features)
        return output.squeeze(-1)