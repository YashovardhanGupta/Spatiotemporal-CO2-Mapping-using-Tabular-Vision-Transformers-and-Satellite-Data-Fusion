# src/5_eda_and_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

def load_and_merge_data(config):
    """Loads and merges the three core datasets."""
    project_dir = Path(__file__).resolve().parents[1]
    
    co2_data_path = project_dir / config['paths']['processed_pkl']
    no2_data_path = project_dir / config['paths']['no2_csv']
    co_data_path = project_dir / config['paths']['co_csv']
    
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
    
    # Rename for clarity in plots
    final_df.rename(columns={'mean_co2_local': 'CO2'}, inplace=True)
    
    print(f"Total aligned data points: {len(final_df)}")
    return final_df

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    with open(project_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    plots_dir = project_dir / config['paths']['plots_dir']
    plots_dir.mkdir(exist_ok=True) # Ensure the plots directory exists

    df = load_and_merge_data(config)

    print("\n--- 1. Data Overview ---")
    print(df[['CO2', 'NO2', 'CO']].describe())

    # --- 2. Time-Series Analysis ---
    print("\n--- 2. Generating Time-Series Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # We need to normalize the data to plot them on a similar scale
    normalized_df = (df[['CO2', 'NO2', 'CO']] - df[['CO2', 'NO2', 'CO']].mean()) / df[['CO2', 'NO2', 'CO']].std()
    
    # Calculate a 30-day rolling average to see trends more clearly
    normalized_df.set_index(df['date']).rolling(window=30).mean().plot(ax=ax)
    
    ax.set_title('Normalized 30-Day Rolling Average of Pollutants (2018-2022)', fontsize=16)
    ax.set_ylabel('Normalized Value (Standard Deviations)')
    ax.set_xlabel('Date')
    ax.legend(title='Pollutant')
    
    plt.tight_layout()
    plot_path = plots_dir / "time_series_trends.png"
    plt.savefig(plot_path)
    print(f"Saved time-series plot to {plot_path}")
    plt.close()

    # --- 3. Distribution Analysis ---
    print("\n--- 3. Generating Distribution Plots ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(data=df, x='CO2', kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Daily CO2 (ppm)')
    
    sns.histplot(data=df, x='NO2', kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Daily NO2 (mol/m^2)')

    sns.histplot(data=df, x='CO', kde=True, ax=axes[2])
    axes[2].set_title('Distribution of Daily CO (mol/m^2)')
    
    plt.tight_layout()
    plot_path = plots_dir / "variable_distributions.png"
    plt.savefig(plot_path)
    print(f"Saved distribution plots to {plot_path}")
    plt.close()
    
    # --- 4. Correlation Analysis ---
    print("\n--- 4. Generating Correlation Heatmap ---")
    correlation_matrix = df[['CO2', 'NO2', 'CO']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Pollutants', fontsize=16)
    
    plot_path = plots_dir / "correlation_heatmap.png"
    plt.savefig(plot_path)
    print(f"Saved correlation heatmap to {plot_path}")
    plt.close()