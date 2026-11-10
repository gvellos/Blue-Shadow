#!/usr/bin/env python3
"""Check a Sentinel-2 L2A zip + DMA AIS csv before writing real loaders.
Usage: python inspect_data.py <S2A_...SAFE.zip | path/to/XXX.SAFE> aisdk-2025-09-27.csv
Needs: pip install rasterio pandas numpy"""
import glob, os, re, sys, zipfile
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform_bounds

s2_in, ais_csv = sys.argv[1], sys.argv[2]
if os.path.isdir(s2_in) and s2_in.rstrip("/").endswith(".SAFE"):
    # already-extracted or fetched SAFE directory (e.g. from fetch_scene.py)
    safe = s2_in.rstrip("/")
else:
    dest = "s2_extracted"
    if not os.path.isdir(dest):
        print("Extracting (about 1 GB, one minute)...")
        zipfile.ZipFile(s2_in).extractall(dest)
    safe = glob.glob(f"{dest}/*.SAFE")[0]
print("SAFE:", os.path.basename(safe))

def band(res, name):
    # downloaded scenes use .jp2; fetch_scene.py writes GeoTIFF (.tif)
    for ext in ("jp2", "tif", "tiff"):
        hits = glob.glob(f"{safe}/GRANULE/*/IMG_DATA/R{res}m/*_{name}_{res}m.{ext}")
        if hits:
            return hits[0]
    return None

# --- time + footprint
m = re.search(r"_(\d{8}T\d{6})_", os.path.basename(safe))
t_img = pd.Timestamp(m.group(1))
b08 = band(10, "B08")
with rasterio.open(b08) as src:
    w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    print(f"B08 10 m: {src.width}x{src.height} px, CRS {src.crs}")
print(f"Image time (UTC): {t_img}")
print(f"Footprint lon {w:.3f}..{e:.3f}, lat {s:.3f}..{n:.3f}")

# --- cloud / water from scene classification (20 m)
scl_path = band(20, "SCL")
if scl_path:
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    tot = scl.size
    frac = lambda cls: 100 * np.isin(scl, cls).sum() / tot
    print(f"Water {frac([6]):.1f}% | cloud {frac([8, 9]):.1f}% | thin cirrus {frac([10]):.1f}% | "
          f"cloud shadow {frac([3]):.1f}% | land/veg/other {100 - frac([6, 8, 9, 10, 3]):.1f}%")

# --- AIS inside footprint, +-90 min around image
t0, t1 = t_img - pd.Timedelta(minutes=90), t_img + pd.Timedelta(minutes=90)
cols = ["# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude", "SOG", "COG", "Length"]
kept = []
print("Reading AIS csv in chunks (can take a minute)...")
for ch in pd.read_csv(ais_csv, usecols=cols, chunksize=1_000_000, low_memory=False):
    ch = ch[ch.Latitude.between(s, n) & ch.Longitude.between(w, e)]
    if len(ch):
        ch = ch.assign(t=pd.to_datetime(ch["# Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"))
        kept.append(ch[(ch.t >= t0) & (ch.t <= t1)])
a = pd.concat(kept) if kept else pd.DataFrame(columns=cols + ["t"])
print(f"AIS messages in footprint +-90 min: {len(a)} | unique MMSI: {a.MMSI.nunique()}")
if len(a):
    near = a[(a.t - t_img).abs() <= pd.Timedelta(minutes=10)]
    print(f"Vessels with a message within +-10 min of image time: {near.MMSI.nunique()}")
    L = a.groupby("MMSI").Length.max()
    print(f"Vessels with a known length: {(L > 0).sum()} / {len(L)}")
    print("Length bands (m):", pd.cut(L[L > 0], [0, 15, 20, 50, 100, 1000]).value_counts().sort_index().to_dict())
    print("Type of mobile:", a.drop_duplicates("MMSI")["Type of mobile"].value_counts().to_dict())
    a.to_csv("ais_window.csv", index=False)
    print("Saved ais_window.csv (filtered AIS)")
