import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("PYROVISION — Data Exploration")
print("=" * 50)

# ── 1. ALGERIA DATASET ──────────────────────────────
print("\n📁 ALGERIA FIRE DATASET")
print("-" * 30)

alg = pd.read_csv("../data/raw/algeria/Algerian_forest_fires_dataset.csv",
                  skiprows=1)          # skip the region header row
alg.columns = alg.columns.str.strip() # remove accidental spaces
print(f"Shape        : {alg.shape}")
print(f"Columns      : {list(alg.columns)}")
print(f"\nFirst 5 rows :")
print(alg.head())
print(f"\nMissing values:\n{alg.isnull().sum()}")

# ── 2. WEATHER DATASET ──────────────────────────────
print("\n\n📁 WEATHER DATASET")
print("-" * 30)

weather_folder = "../data/raw/weather"
weather_files  = os.listdir(weather_folder)
print(f"Files found  : {weather_files}")

# read whichever CSV is in there
csv_files = [f for f in weather_files if f.endswith('.csv')]
if csv_files:
    wx = pd.read_csv(f"{weather_folder}/{csv_files[0]}")
    print(f"Shape        : {wx.shape}")
    print(f"Columns      : {list(wx.columns)}")
    print(f"\nFirst 5 rows :")
    print(wx.head())
else:
    print("No CSV found — check folder")

# ── 3. WILDFIRE SQLITE ───────────────────────────────
print("\n\n📁 WILDFIRE DATABASE")
print("-" * 30)

import sqlite3
db_path = "../data/raw/viirs/FPA_FOD_20170508.sqlite"
conn    = sqlite3.connect(db_path)

# see what tables exist
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(f"Tables inside: {list(tables['name'])}")

# peek at the main fire table
fires = pd.read_sql("SELECT * FROM Fires LIMIT 5", conn)
print(f"\nColumns: {list(fires.columns)}")
print(f"\nSample rows:")
print(fires[['FIRE_YEAR','DISCOVERY_DATE','FIRE_SIZE',
             'LATITUDE','LONGITUDE','STAT_CAUSE_DESCR']].head())

conn.close()

# ── 4. DEM ELEVATION ────────────────────────────────
print("\n\n📁 DEM ELEVATION FILES")
print("-" * 30)

dem_folder = "../data/raw/dem"
dem_files  = os.listdir(dem_folder)
tif_files  = [f for f in dem_files if f.endswith('.tif') or f.endswith('.TIF')]
print(f"TIF files found: {tif_files}")

if tif_files:
    import rasterio
    with rasterio.open(f"{dem_folder}/{tif_files[0]}") as src:
        print(f"\nFile         : {tif_files[0]}")
        print(f"Resolution   : {src.res} degrees per pixel")
        print(f"CRS          : {src.crs}")
        print(f"Grid size    : {src.width} x {src.height} pixels")
        print(f"Bounds       : {src.bounds}")
        data = src.read(1)
        print(f"Elevation min: {data.min()} m")
        print(f"Elevation max: {data.max()} m")

# ── 5. LULC ─────────────────────────────────────────
print("\n\n📁 LULC LAND COVER")
print("-" * 30)

lulc_folder = "../data/raw/lulc"
all_items   = os.listdir(lulc_folder)
print(f"Contents: {all_items}")

# count images inside subfolders
for item in all_items:
    full_path = f"{lulc_folder}/{item}"
    if os.path.isdir(full_path):
        sub = os.listdir(full_path)
        print(f"  Folder '{item}' has {len(sub)} items")
        # go one level deeper
        for sub_item in sub[:5]:
            sub_path = f"{full_path}/{sub_item}"
            if os.path.isdir(sub_path):
                count = len(os.listdir(sub_path))
                print(f"    └── '{sub_item}' → {count} images")

print("\n\n✅ Exploration complete")
print("=" * 50)