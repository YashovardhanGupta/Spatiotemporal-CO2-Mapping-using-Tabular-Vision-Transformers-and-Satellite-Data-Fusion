# src/4_evaluate.py (v2 - MLP Update)

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json

# We need the class definition for our dataset to load the dataloader file
from dataset import SpatiotemporalDataset 
# We also need the model component definitions to rebuild the architecture
from einops.layers.torch import Rearrange


# --- Model architectures must match the trained model ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, emb_size=1024, image_size=64):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (image_size // patch_size) ** 2
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x

class VisionTransformer(nn.Module):
    def __init__(self, emb_size=1024, depth=6, heads=16, mlp_dim=2048, **kwargs):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.patch_embed = PatchEmbedding(emb_size=emb_size, **kwargs)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return x[:, 0]

class TabularMLP(nn.Module):
    """The MLP model for our new continuous tabular data (NO2, CO)."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)

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

# --- Main Evaluation Function ---
def evaluate_model():
    project_dir = Path(__file__).resolve().parents[1]
    with open(project_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    dataloaders_path = project_dir / "data_processed" / "dataloaders.pth"
    dataloaders = torch.load(dataloaders_path, weights_only=False)
    test_loader = dataloaders['test']
    
    sample_batch = next(iter(test_loader))
    tabular_input_dim = sample_batch['tabular'].shape[1]

    # --- Reconstruct the CORRECT Model Architecture (with TabularMLP) ---
    vit_config = config['model']
    vit = VisionTransformer(
        in_channels=1, patch_size=vit_config['patch_size_vit'], emb_size=vit_config['dim'],
        image_size=vit_config['image_size'], depth=vit_config['depth'],
        heads=vit_config['heads'], mlp_dim=vit_config['mlp_dim']
    )
    TABULAR_OUTPUT_DIM = 64
    tabular_mlp = TabularMLP(input_dim=tabular_input_dim, hidden_dims=[128, 96], output_dim=TABULAR_OUTPUT_DIM)
    
    model = SpatiotemporalFusionModel(
        vit_model=vit, tabular_model=tabular_mlp,
        vit_output_dim=vit_config['dim'], tabular_output_dim=TABULAR_OUTPUT_DIM
    ).to(device)

    # --- Load the Saved Weights ---
    model_path = project_dir / config['paths']['checkpoints_dir'] / "best_fusion_model.pth"
    print(f"Loading best model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Run Predictions ---
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images, tabular, targets = batch['image'].to(device), batch['tabular'].to(device), batch['target'].to(device)
            predictions = model(images, tabular)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # --- Calculate and Print Metrics ---
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    print("\n--- Final Model Performance on Test Set ---")
    print(f"Mean Squared Error (MSE):      {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.6f}")
    print(f"Mean Absolute Error (MAE):       {mae:.6f}")
    print(f"R-squared (R²):                  {r2:.4f}")
    print("-------------------------------------------")
    
    metrics = {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
    metrics_path = project_dir / "results" / "test_metrics.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate_model()