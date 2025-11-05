# src/3_train.py (v6 - Refactored, GPU-Enabled)

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import time
import copy

# Import our modularized components
from dataset import SpatiotemporalDataset
from models.vision_transformer import VisionTransformer
from models.tabular_mlp import TabularMLP
from models.fusion_model import SpatiotemporalFusionModel

# The train_model function is unchanged
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, checkpoint_path):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}'); print('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            for batch in dataloaders[phase]:
                images, tabular, targets = batch['image'].to(device), batch['tabular'].to(device), batch['target'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, tabular)
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.6f}')
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✅ New best model saved to {checkpoint_path}")
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Loss: {best_val_loss:6f}')
    model.load_state_dict(best_model_wts)
    return model

# The main execution block
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    with open(project_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- THIS IS THE GPU-AWARE VERSION ---
    # This line automatically checks for a GPU and uses it if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- END OF CHANGE ---
    
    print(f"Using device: {device}")
    
    dataloaders_path = project_dir / "data_processed" / "dataloaders.pth"
    dataloaders = torch.load(dataloaders_path, weights_only=False)
    
    checkpoint_path = project_dir / config['paths']['checkpoints_dir']
    checkpoint_path.mkdir(exist_ok=True)
    best_model_path = checkpoint_path / "best_fusion_model.pth"

    sample_batch = next(iter(dataloaders['train']))
    tabular_input_dim = sample_batch['tabular'].shape[1]
    
    vit_config = config['model']
    vit = VisionTransformer(
        in_channels=1, patch_size=vit_config['patch_size_vit'], emb_size=vit_config['dim'],
        image_size=vit_config['image_size'], depth=vit_config['depth'],
        heads=vit_config['heads'], mlp_dim=vit_config['mlp_dim']
    )
    
    TABULAR_OUTPUT_DIM = 64
    tabular_mlp = TabularMLP(input_dim=tabular_input_dim, hidden_dims=[128, 96], output_dim=TABULAR_OUTPUT_DIM)
    
    fusion_model = SpatiotemporalFusionModel(
        vit_model=vit, tabular_model=tabular_mlp,
        vit_output_dim=vit_config['dim'], tabular_output_dim=TABULAR_OUTPUT_DIM
    ).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=config['training']['learning_rate'])
    
    print("\nStarting model training...")
    train_model(
        model=fusion_model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
        device=device, num_epochs=config['training']['epochs'], checkpoint_path=best_model_path
    )