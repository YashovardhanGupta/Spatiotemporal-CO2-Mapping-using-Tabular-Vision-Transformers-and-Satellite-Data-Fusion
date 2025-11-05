# Spatiotemporal Data Fusion for CO₂ Prediction

This project is an end-to-end deep learning pipeline designed to predict daily CO₂ concentrations at specific industrial sites. It uses a spatiotemporal data fusion approach, combining 2D satellite imagery with 1D time-series data.

The model was built in PyTorch and leverages a two-branch neural network to learn from these different data types simultaneously.

## Core Methodology

The goal is to predict the mean CO₂ concentration for the next day (T+1) at a given location. To do this, the model is fed two types of data from the current day (T):

1. Spatial Data (ViT Branch): 1°x1° image patches of CO₂ (from NASA's OCO-2 satellite) centered on the location. This allows the model to learn spatial patterns.

2. Temporal Data (MLP Branch): 1D vector of proxy pollutants `[NO₂, CO]` (from Sentinel-5P) for that day. This provides a signal for local industrial activity.

The insights from both branches are then fused to make a single predictive forecast.

## Key Scientific Finding

A major part of this project was an Exploratory Data Analysis (EDA) which revealed that the direct daily correlation between CO₂ and the proxy gases (NO₂ and CO) was very weak (r < 0.2). While the model pipeline was successfully built, this weak underlying relationship was the primary constraint on predictive performance. This is a key finding: this specific data fusion hypothesis is not strongly supported by the data, suggesting future work should focus on incorporating other data sources, such as meteorological data.

## Project Structure

    Spatiotemporal_CO2_Mapping_using_Tabular_Vision_Transformers_and_Satellite_Data_Fusion/
    │
    ├── data_raw/
    │   ├── netcdf/                 # Contains all raw .nc CO₂ files
    │   ├── Cleaned_Location_Details.csv # Master list of plant locations
    │   ├── Daily_CO_Time_Series_Data.csv  # Raw CO data from GEE
    │   └── Daily_NO2_Time_Series_Data.csv # Raw NO₂ data from GEE
    │
    ├── data_processed/
    │   ├── processed_data.pkl      # Output of script 1 (CO₂ images + means)
    │   └── dataloaders.pth         # Output of script 2 (Final PyTorch DataLoaders)
    │
    ├── results/
    │   ├── checkpoints/
    │   │   └── best_fusion_model.pth # The final trained model
    │   └── plots/
    │       ├── correlation_heatmap.png
    │       ├── time_series_trends.png
    │       └── variable_distributions.png
    │
    ├── src/
    │   ├── models/                 # Contains model component scripts
    │   │   ├── __init__.py
    │   │   ├── fusion_model.py
    │   │   ├── tabular_mlp.py
    │   │   └── vision_transformer.py
    │   │
    │   ├── 1_preprocess_netcdf.py  # Script 1: Processes raw NetCDF files
    │   ├── 2_dataset_creator.py    # Script 2: Fuses all data sources into DataLoaders
    │   ├── 3_train.py              # Script 3: Trains the fusion model
    │   ├── 4_evaluate.py           # Script 4: Evaluates the trained model (optional)
    │   └── 5_eda_and_visualization.py # Script 5: Generates plots for the report
    │
    ├── config.yaml               # All settings (paths, model params, training)
    ├── requirements.txt          # All required Python packages
    └── README.md                 # This file


## How to Run the Project

1. Setup

First, install all required dependencies from your virtual environment.

    pip install -r requirements.txt


2. Run the Full Pipeline

The entire process is broken into sequential scripts. You must run them in order.

**Step 1: Process Raw CO₂ Data**
This script reads all raw ``.nc`` files from ``data_raw/netcdf/``, extracts the 1°x1° image patches for each plant, applies the scientific scale factor, and saves the output to ``data_processed/processed_data.pkl``.

    python src/1_preprocess_netcdf.py


**Step 2: Create the Fused Dataset**
This script reads the `processed_data.pkl` (from Step 1) and the two proxy gas CSVs (`Daily_CO...` and `Daily_NO2...`). It performs a three-way inner merge to align all data by `plant_name` and `date`. It then splits the data (Train/Val/Test), applies normalization, and saves the final PyTorch `DataLoader` objects to `data_processed/dataloaders.pth`.

    python src/2_dataset_creator.py


**Step 3: Train the Model**
This script loads the `dataloaders.pth` (from Step 2), builds the ViT, MLP, and Fusion models, and begins the training process. It will automatically use a GPU if `torch.cuda.is_available()` is true. The best-performing model (based on validation loss) will be saved to `results/checkpoints/best_fusion_model.pth`.

    python src/3_train.py


**Step 4: (Optional) Run EDA & Evaluation**
After training, you can run the EDA script to generate the plots and a final evaluation script (if built) to get test set metrics.

    python src/5_eda_and_visualization.py
