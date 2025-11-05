# src/0_download_data.py (v3 - Final)

import yaml
from pathlib import Path
import earthaccess
import sys

def main():
    """
    Main function to authenticate with NASA Earthdata and download .nc files.
    This version uses a more robust authentication strategy.
    """
    # --- 1. Load Configuration ---
    try:
        project_dir = Path(__file__).resolve().parents[1]
        with open(project_dir / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: config.yaml not found. Make sure you are running the script from the 'src' directory.")
        sys.exit(1) # Exit the script if config is not found

    links_dir = project_dir / config['paths']['netcdf_links_dir']
    download_dir = project_dir / config['paths']['raw_netcdf_dir']

    download_dir.mkdir(exist_ok=True)
    
    # --- 2. Robust Authentication ---
    print("Authenticating with NASA Earthdata Login...")
    try:
        # The `persist=True` flag will save your credentials after a successful
        # login, so you don't have to log in every time.
        # It will automatically try .netrc first, then open a browser if needed.
        earthaccess.login(strategy="interactive", persist=True)
    except Exception as e:
        print(f"❌ An error occurred during authentication: {e}")
        print("Please try running the script again.")
        sys.exit(1)
        
    print("✅ Authentication successful.")

    # --- 3. Find and Read Link Files ---
    txt_files = sorted(list(links_dir.glob("*.txt")))
    if not txt_files:
        print(f"❌ No .txt files found in {links_dir}. Please place your URL list files there.")
        sys.exit(1)

    all_urls = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip().startswith('http')]
            all_urls.extend(urls)
    
    # --- 4. Download Files ---
    if not all_urls:
        print("No URLs found in the text files. Nothing to download.")
        sys.exit(0)
        
    print(f"\nFound {len(all_urls)} URLs. Starting download to: {download_dir}")
    
    try:
        earthaccess.download(all_urls, local_path=str(download_dir), threads=8)
    except Exception as e:
        print(f"\n❌ An error occurred during the download process: {e}")
        print("Some files may not have been downloaded. You can try running the script again to resume.")
        sys.exit(1)
    
    print("\n--- ✅ All files downloaded successfully! ---")


if __name__ == "__main__":
    main()