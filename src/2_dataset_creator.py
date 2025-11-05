# src/2_dataset_creator.py (v8 - Multiprocessing Fix)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import yaml
from pathlib import Path
import torchvision.transforms as T
import joblib

# The class is defined at the top level
class SpatiotemporalDataset(Dataset):
    def __init__(self, dataframe, tabular_features, image_transform=None):
        self.df = dataframe
        self.image_transform = image_transform
        self.tabular_features = tabular_features
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_patch = row['co2_patch'].astype(np.float32)
        image_patch = np.expand_dims(image_patch, axis=0)
        if self.image_transform:
            image_patch = self.image_transform(torch.from_numpy(image_patch))
        tabular_data = row[self.tabular_features].values.astype(np.float32)
        target = row['target'].astype(np.float32)
        return {
            'image': image_patch,
            'tabular': torch.from_numpy(tabular_data),
            'target': torch.tensor(target, dtype=torch.float32)
        }

def create_dataloaders(config_path):
    # This function now only contains the logic, it doesn't run itself.
    project_dir = Path(__file__).resolve().parents[1]
    # ... (rest of the function is the same as before) ...
    with open(project_dir / config_path, "r") as f:
        config = yaml.safe_load(f)
    co2_data_path = project_dir / config['paths']['processed_pkl']
    no2_data_path = project_dir / config['paths']['no2_csv']
    co_data_path = project_dir / config['paths']['co_csv']
    scaler_path = project_dir / config['paths']['scaler_pkl']
    print("Loading 3 core datasets (CO2, NO2, CO)...")
    df_co2 = pd.read_pickle(co2_data_path)
    df_no2 = pd.read_csv(no2_data_path)
    df_co = pd.read_csv(co_data_path)
    print("Preparing and cleaning data for merge...")
    for df in [df_co2, df_no2, df_co]:
        df['date'] = pd.to_datetime(df['date'])
    df_no2.rename(columns={'mean': 'NO2', 'name': 'plant_name'}, inplace=True)
    df_co.rename(columns={'mean': 'CO', 'name': 'plant_name'}, inplace=True)
    df_no2 = df_no2[['plant_name', 'date', 'NO2']]
    df_co = df_co[['plant_name', 'date', 'CO']]
    print("Performing 3-way inner merge to align all datasets...")
    merged_df = pd.merge(df_co2, df_no2, on=['plant_name', 'date'], how='inner')
    final_df = pd.merge(merged_df, df_co, on=['plant_name', 'date'], how='inner')
    final_df.sort_values(by=['plant_name', 'date'], inplace=True)
    print(f"Total aligned data points after merge: {len(final_df)}")
    print("Creating target variable (next day's CO2 reading)...")
    final_df['target'] = final_df.groupby('plant_name')['mean_co2_local'].shift(-1)
    final_df.dropna(subset=['target'], inplace=True)
    tabular_features = ['NO2', 'CO']
    print("Splitting data and normalizing tabular features...")
    final_df['year'] = final_df['date'].dt.year
    train_df = final_df[final_df['year'] <= 2020].copy()
    val_df = final_df[final_df['year'] == 2021].copy()
    test_df = final_df[final_df['year'] >= 2022].copy()
    scaler = StandardScaler()
    train_df.loc[:, tabular_features] = scaler.fit_transform(train_df[tabular_features])
    val_df.loc[:, tabular_features] = scaler.transform(val_df[tabular_features])
    test_df.loc[:, tabular_features] = scaler.transform(test_df[tabular_features])
    image_transform = T.Compose([
        T.Resize((config['model']['image_size'], config['model']['image_size']), antialias=True),
        T.Normalize(mean=[final_df['mean_co2_local'].mean()], std=[final_df['mean_co2_local'].std()])
    ])
    train_dataset = SpatiotemporalDataset(train_df, tabular_features, image_transform)
    val_dataset = SpatiotemporalDataset(val_df, tabular_features, image_transform)
    test_dataset = SpatiotemporalDataset(test_df, tabular_features, image_transform)
    
    # --- THIS IS THE KEY CHANGE ---
    # Set num_workers to a number like 4 to enable parallel data loading.
    # A good rule of thumb is half the number of your CPU cores.
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    output_path = project_dir / "data_processed" / "dataloaders.pth"
    torch.save(dataloaders, output_path)
    joblib.dump(scaler, scaler_path)
    print("-" * 50)
    print(f"✅ Success! DataLoaders saved to: {output_path}")
    print(f"✅ Success! Tabular data scaler saved to: {scaler_path}")

# --- THIS IS THE OTHER KEY CHANGE ---
# We wrap the main call in this block to ensure multiprocessing safety on Windows.
if __name__ == "__main__":
    create_dataloaders("config.yaml")