# src/1_preprocess_netcdf.py (v6 - Scale Factor Fix)

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import yaml
import warnings
import sys

# ... (imports and get_main_data_variable function are the same) ...
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_main_data_variable(ds):
    for var_name, data_array in ds.data_vars.items():
        if 'lat' in data_array.dims and 'lon' in data_array.dims:
            return var_name
    raise ValueError("Could not automatically determine the main data variable.")

def extract_co2_data(netcdf_path, plant_lat, plant_lon, patch_size_deg):
    try:
        lat_min, lat_max = plant_lat - patch_size_deg / 2, plant_lat + patch_size_deg / 2
        lon_min, lon_max = plant_lon - patch_size_deg / 2, plant_lon + patch_size_deg / 2

        # Use mask_and_scale=True to let xarray handle scaling if possible
        with xr.open_dataset(netcdf_path, mask_and_scale=True) as ds:
            main_var_name = get_main_data_variable(ds)
            
            # This is the raw data selection
            patch_data = ds[main_var_name].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            
            if patch_data.ndim > 2:
                patch_data = patch_data.squeeze(drop=True)

            if patch_data.size == 0 or np.all(np.isnan(patch_data.values)):
                return None, None
            
            mean_co2 = float(np.nanmean(patch_data.values))
            
            # Use the already-scaled data for the patch
            co2_patch_filled = patch_data.fillna(mean_co2).values
            
            if co2_patch_filled.ndim == 0:
                co2_patch_filled = np.array([[co2_patch_filled]])
            elif co2_patch_filled.ndim == 1:
                co2_patch_filled = np.expand_dims(co2_patch_filled, axis=0)
            
            # mean_co2 is already calculated from the scaled data
            return co2_patch_filled, mean_co2
            
    except Exception:
        return None, None

# ... (The main() function is exactly the same as before) ...
def main():
    project_dir = Path(__file__).resolve().parents[1]
    with open(project_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    locations_file = project_dir / config['paths']['locations_csv']
    netcdf_dir = project_dir / config['paths']['raw_netcdf_dir']
    output_file = project_dir / config['paths']['processed_pkl']
    patch_size = config['data']['patch_size_deg']
    output_file.parent.mkdir(exist_ok=True)
    print("Loading plant locations...")
    plants_df = pd.read_csv(locations_file)
    print(f"Found {len(plants_df)} plant locations.")
    netcdf_files = sorted(list(netcdf_dir.glob("*.nc")))
    if not netcdf_files:
        print(f"❌ Error: No NetCDF files found in {netcdf_dir}")
        sys.exit(1)
    print(f"Found {len(netcdf_files)} NetCDF files to process.")
    all_extracted_data = []
    for index, plant in plants_df.iterrows():
        plant_name = plant['name']
        plant_lat = plant['latitude']
        plant_lon = plant['longitude']
        print(f"\nProcessing data for: {plant_name}")
        for nc_file in netcdf_files:
            try:
                date_str = nc_file.stem.split('.')[-2] 
                date_obj = pd.to_datetime(date_str, format='%Y%m%d')
            except (IndexError, ValueError):
                print(f"  -> Could not parse date from filename: {nc_file.name}")
                continue
            co2_patch, mean_co2 = extract_co2_data(nc_file, plant_lat, plant_lon, patch_size)
            if co2_patch is not None:
                record = {
                    "plant_name": plant_name, "latitude": plant_lat, "longitude": plant_lon,
                    "date": date_obj, "mean_co2_local": mean_co2, "co2_patch": co2_patch
                }
                all_extracted_data.append(record)
    if not all_extracted_data:
        print("-" * 50); print("❌ Error: No data was extracted."); sys.exit(1)
    print("\nConsolidating all data...")
    final_df = pd.DataFrame(all_extracted_data)
    final_df.to_pickle(output_file)
    print("-" * 50)
    print(f"✅ Success! Processed data saved to: {output_file}")
    print("Here's a sample of your new, unified dataset:")
    print(final_df[['plant_name', 'date', 'mean_co2_local']].head())
    print(f"\nTotal data points extracted: {len(final_df)}")

if __name__ == "__main__":
    main()