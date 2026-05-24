import os
import urllib.request
import pandas as pd
import numpy as np

# Dataset URLs (FY 2026 / PY 2026 data catalog)
HRRP_URL = "https://data.cms.gov/provider-data/sites/default/files/resources/a171bc36c488d3e0dc33ec63abb469a6_1770163617/FY_2026_Hospital_Readmissions_Reduction_Program_Hospital.csv"
INFO_URL = "https://data.cms.gov/provider-data/sites/default/files/resources/893c372430d9d71a1c52737d01239d47_1777413958/Hospital_General_Information.csv"

DATA_DIR = "data"
HRRP_CACHE = os.path.join(DATA_DIR, "Hospital_Readmissions_Reduction_Program.csv")
INFO_CACHE = os.path.join(DATA_DIR, "Hospital_General_Information.csv")
MERGED_CACHE = os.path.join(DATA_DIR, "merged_cms_data.csv")

def download_file(url, destination):
    """Downloads a file with basic browser-like user agent to avoid bot blockers."""
    print(f"Downloading {url} to {destination}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with open(destination, 'wb') as f:
            f.write(response.read())
    print("Download completed.")

def preprocess_and_merge():
    """Downloads, cleans, and merges CMS datasets, saving results to local CSV cache."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Download datasets if not cached
    if not os.path.exists(HRRP_CACHE):
        download_file(HRRP_URL, HRRP_CACHE)
    else:
        print(f"Using cached HRRP dataset: {HRRP_CACHE}")

    if not os.path.exists(INFO_CACHE):
        download_file(INFO_URL, INFO_CACHE)
    else:
        print(f"Using cached General Info dataset: {INFO_CACHE}")

    # 2. Load dataframes
    print("Loading datasets into Pandas...")
    hrrp_df = pd.read_csv(HRRP_CACHE, dtype=str)
    info_df = pd.read_csv(INFO_CACHE, dtype=str)

    # 3. Clean HRRP dataset
    print("Cleaning HRRP dataset...")
    hrrp_df['Facility ID'] = hrrp_df['Facility ID'].astype(str).str.zfill(6)
    
    numeric_cols = [
        'Number of Discharges', 
        'Excess Readmission Ratio', 
        'Predicted Readmission Rate', 
        'Expected Readmission Rate', 
        'Number of Readmissions'
    ]
    for col in numeric_cols:
        if col in hrrp_df.columns:
            # Clean string symbols like commas, N/A, and low counts text
            cleaned_series = hrrp_df[col].astype(str).str.replace(',', '').str.strip()
            cleaned_series = cleaned_series.replace(['N/A', 'Too Few to Report', 'Too few to report', ''], np.nan)
            hrrp_df[col] = pd.to_numeric(cleaned_series, errors='coerce')

    # 4. Clean General Information dataset
    print("Cleaning Hospital General Info dataset...")
    info_df['Facility ID'] = info_df['Facility ID'].astype(str).str.zfill(6)
    
    # We'll merge specific columns from General Info:
    # Facility ID, Address, City/Town, ZIP Code, County/Parish, Hospital Type, Hospital Ownership, Emergency Services, Hospital overall rating
    info_cols = [
        'Facility ID', 
        'Address', 
        'City/Town', 
        'ZIP Code', 
        'County/Parish', 
        'Hospital Type', 
        'Hospital Ownership', 
        'Emergency Services', 
        'Hospital overall rating'
    ]
    info_subset = info_df[[c for c in info_cols if c in info_df.columns]].copy()
    
    # Clean rating to numeric
    if 'Hospital overall rating' in info_subset.columns:
        cleaned_rating = info_subset['Hospital overall rating'].astype(str).str.replace('Not Available', '').str.strip()
        cleaned_rating = cleaned_rating.replace(['N/A', ''], np.nan)
        info_subset['Hospital overall rating'] = pd.to_numeric(cleaned_rating, errors='coerce')

    # 5. Merge
    print("Merging datasets on Facility ID...")
    merged_df = pd.merge(hrrp_df, info_subset, on='Facility ID', how='left')

    # Save to cache
    merged_df.to_csv(MERGED_CACHE, index=False)
    print(f"Preprocessing done. Merged dataset saved to {MERGED_CACHE} (Shape: {merged_df.shape})")
    return merged_df, hrrp_df, info_subset

def load_data(force_reload=False):
    """Loads datasets, downloading and preprocessing them if missing or forced."""
    if force_reload or not os.path.exists(MERGED_CACHE) or not os.path.exists(HRRP_CACHE) or not os.path.exists(INFO_CACHE):
        print("Merged dataset cache missing or reload forced. Running download & preprocessing...")
        return preprocess_and_merge()
    
    print("Loading datasets from local CSV cache...")
    merged_df = pd.read_csv(MERGED_CACHE, dtype={'Facility ID': str})
    hrrp_df = pd.read_csv(HRRP_CACHE, dtype={'Facility ID': str})
    info_df = pd.read_csv(INFO_CACHE, dtype={'Facility ID': str})
    
    # Ensure numeric columns are properly parsed in merged_df
    numeric_cols = [
        'Number of Discharges', 
        'Excess Readmission Ratio', 
        'Predicted Readmission Rate', 
        'Expected Readmission Rate', 
        'Number of Readmissions',
        'Hospital overall rating'
    ]
    for col in numeric_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        if col in hrrp_df.columns:
            hrrp_df[col] = pd.to_numeric(hrrp_df[col], errors='coerce')
            
    return merged_df, hrrp_df, info_df

if __name__ == "__main__":
    # Test script directly
    load_data(force_reload=True)
