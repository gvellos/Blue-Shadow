#!/usr/bin/env python3
"""Fetch a real Sentinel-2 L2A scene from Microsoft Planetary Computer by date + bounding box.

No manual 1 GB SAFE downloads: this queries the Planetary Computer STAC API (no credentials needed), picks the
least-cloudy scene over your area of interest in the date window, and reads ONLY the bands and the bbox window
pipeline_real.py needs (B08 + B02/B03/B04 at 10 m, SCL at 20 m) straight from the Cloud-Optimized GeoTIFFs via
HTTP range requests. It writes a small SAFE-like folder (GeoTIFF instead of JP2 -- the pipeline reads it the same
way) with an MTD_TL.xml carrying the sensing time.

Usage:
  python fetch_scene.py --bbox 11.80 54.92 13.59 55.94 --date 2025-09-27
  python fetch_scene.py --bbox 11.80 54.92 13.59 55.94 --date 2025-09-27 --days 6 --max-cloud 20 --out s2_extracted
then:
  OUT=./out python pipeline_real.py --safe s2_extracted/<printed>.SAFE --ais ais_window.csv

Tip: get AIS for the same scene/day with inspect_data.py (it writes ais_window.csv). bbox is lon/lat:
minLon minLat maxLon maxLat.

Sentinel-1 radar (works at night and through cloud), from the sentinel-1-rtc collection:
  python fetch_scene.py --sensor s1 --bbox 11.80 54.92 13.59 55.94 --date 2025-11-20 --night
writes s1_extracted/<id>.S1/ (vv.tif, vh.tif, landmask.tif from ESA WorldCover, scene.json); then
  OUT=./out_s1 python pipeline_sar.py --scene s1_extracted/<printed>.S1 --ais ais_window.csv --osm
For an ICEYE product (bought or tasked), only the land mask is needed from here:
  python fetch_scene.py --landmask-for path/to/ICEYE_..._GRD.tif

Requires: pip install pystac-client planetary-computer rasterio
"""
import argparse, os, sys
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds, bounds as win_bounds

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS_10M = ("B08", "B04", "B03", "B02")      # NIR + true-colour; pipeline_real needs all four
BAND_20M = "SCL"                               # scene classification -> land/cloud/water masks


# --------------------------------------------------------------------- STAC search
def _bbox_overlap_frac(item, bbox):
    """Fraction of the query bbox covered by the item footprint, if shapely is available (else 1.0)."""
    try:
        from shapely.geometry import box, shape
    except Exception:
        return 1.0
    aoi = box(*bbox)
    if aoi.area == 0:
        return 1.0
    try:
        return shape(item.geometry).intersection(aoi).area / aoi.area
    except Exception:
        return 1.0


def _nodata_pct(item):
    for k in ("s2:nodata_pixel_percentage", "s2:degraded_msi_data_percentage"):
        v = item.properties.get(k)
        if v is not None:
            return float(v)
    dc = item.properties.get("data_coverage")        # some catalogues report coverage instead
    return 100.0 - float(dc) if dc is not None else 0.0


def search_scene(bbox, date, days, max_cloud, min_coverage):
    import planetary_computer as pc
    from pystac_client import Client

    d0 = datetime.strptime(date, "%Y-%m-%d")
    lo, hi = d0 - timedelta(days=days), d0 + timedelta(days=days)
    dt = f"{lo:%Y-%m-%d}/{hi:%Y-%m-%d}"
    cat = Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = cat.search(collections=[COLLECTION], bbox=bbox, datetime=dt,
                        query={"eo:cloud_cover": {"lt": max_cloud}})
    items = list(search.items())
    if not items:
        sys.exit(f"No {COLLECTION} scene over bbox in {dt} under {max_cloud}% cloud. "
                 f"Widen --days or raise --max-cloud.")

    scored = []
    for it in items:
        scored.append(dict(item=it, nodata=_nodata_pct(it), cloud=it.properties.get("eo:cloud_cover", 100),
                           overlap=_bbox_overlap_frac(it, bbox),
                           ddays=abs((it.datetime.replace(tzinfo=None) - d0).days)))
    # drop near-empty swath-edge tiles (mostly nodata) and ones that barely touch the bbox
    good = [s for s in scored if s["nodata"] <= (100 - min_coverage) and s["overlap"] >= 0.15]
    pool = good or scored
    # prefer: closest date -> most bbox overlap -> least nodata -> least cloud
    pool.sort(key=lambda s: (s["ddays"], -round(s["overlap"], 2), s["nodata"], s["cloud"]))
    if not good:
        print(f"WARNING: every candidate is <{min_coverage}% data over this bbox (swath-edge tiles). "
              f"Using the best available; consider a bbox nearer the tile centre or a different --date.")
    best = pool[0]
    print(f"{len(items)} candidate scene(s); chose {best['item'].id}")
    print(f"  {best['nodata']:.0f}% nodata, {best['cloud']}% cloud, {best['overlap']*100:.0f}% of bbox covered, "
          f"{best['item'].datetime:%Y-%m-%d %H:%M} UTC")
    items = [best["item"]]
    return items[0]


# --------------------------------------------------------------------- windowed assembly
def _snap_window(src, bounds_utm):
    """Pixel window for bounds, snapped to EVEN offsets/sizes so the 20 m SCL (half resolution) stays aligned
    to the 10 m grid (one 20 m pixel == a 2x2 block of 10 m pixels sharing the tile origin)."""
    w = from_bounds(*bounds_utm, transform=src.transform).round_offsets().round_lengths()
    r0 = int(w.row_off) - (int(w.row_off) & 1)
    c0 = int(w.col_off) - (int(w.col_off) & 1)
    h = (int(w.height) + 1) & ~1
    wd = (int(w.width) + 1) & ~1
    r0, c0 = max(0, r0), max(0, c0)
    h = min(h, src.height - r0)
    wd = min(wd, src.width - c0)
    return Window(c0, r0, wd & ~1, h & ~1)


def _write(path, data, src, transform):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                       dtype=str(data.dtype), crs=src.crs, transform=transform, compress="deflate") as dst:
        dst.write(data, 1)


def assemble(assets, bbox_lonlat, out_dir, scene_id, sensing_iso, open_fn=rasterio.open):
    """Read the bbox window of each band from its (signed) href and write a SAFE-like tree. `open_fn` is injected so
    this is testable offline against local GeoTIFFs. Returns the .SAFE path."""
    safe = f"{out_dir}/{scene_id}.SAFE"
    gran = f"{safe}/GRANULE/L2A_{scene_id[:60]}"
    os.makedirs(gran, exist_ok=True)

    # anchor everything to the snapped 10 m B08 window, then derive the exact same geographic box for the others
    with open_fn(assets["B08"]) as b08:
        box = transform_bounds("EPSG:4326", b08.crs, *bbox_lonlat)
        w10 = _snap_window(b08, box)
        if w10.width < 16 or w10.height < 16:
            sys.exit("bbox does not overlap the chosen scene (empty window). Check lon/lat order: minLon minLat maxLon maxLat.")
        geo = win_bounds(w10, b08.transform)                       # UTM bounds shared by all bands
        a08 = b08.read(1, window=w10)
        valid_frac = float((a08 > 0).mean())                        # nonzero = real data (0 = swath-edge nodata)
        _write(f"{gran}/IMG_DATA/R10m/{scene_id}_B08_10m.tif", a08, b08, b08.window_transform(w10))

    for b in BANDS_10M:
        if b == "B08":
            continue
        with open_fn(assets[b]) as src:
            w = _snap_window(src, geo)
            _write(f"{gran}/IMG_DATA/R10m/{scene_id}_{b}_10m.tif", src.read(1, window=w), src, src.window_transform(w))

    with open_fn(assets[BAND_20M]) as scl:
        w = from_bounds(*geo, transform=scl.transform).round_offsets().round_lengths()
        w = Window(int(w.col_off), int(w.row_off), int(w.width), int(w.height))
        _write(f"{gran}/IMG_DATA/R20m/{scene_id}_SCL_20m.tif", scl.read(1, window=w), scl, scl.window_transform(w))

    with open(f"{gran}/MTD_TL.xml", "w") as f:
        f.write(f'<n1:Level-2A_Tile_ID xmlns:n1="x"><General_Info><SENSING_TIME metadataLevel="Standard">'
                f'{sensing_iso}</SENSING_TIME></General_Info></n1:Level-2A_Tile_ID>')
    return safe, valid_frac


# --------------------------------------------------------------------- Sentinel-1 (radar: night + cloud)
S1_COLLECTION = "sentinel-1-rtc"          # terrain-corrected gamma0, linear power, UTM 10 m COGs (assets vv / vh)
WORLDCOVER = "esa-worldcover"             # 10 m land cover; 80 = permanent water, 0 = no data


def search_s1(bbox, date, days, night):
    import planetary_computer as pc
    from pystac_client import Client
    import sar as S

    d0 = datetime.strptime(date, "%Y-%m-%d")
    dt = f"{d0 - timedelta(days=days):%Y-%m-%d}/{d0 + timedelta(days=days):%Y-%m-%d}"
    cat = Client.open(PC_STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=[S1_COLLECTION], bbox=bbox, datetime=dt).items())
    if not items:
        sys.exit(f"No {S1_COLLECTION} scene over bbox in {dt}. Widen --days.")
    lon_c, lat_c = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    scored = []
    for it in items:
        t = it.datetime.replace(tzinfo=None)
        sun = S.sun_elevation_deg(lat_c, lon_c, t)
        scored.append(dict(item=it, sun=sun, is_night=sun < -6, overlap=_bbox_overlap_frac(it, bbox),
                           ddays=abs((t - d0).days), orbit=it.properties.get("sat:orbit_state", "?")))
    pool = [s for s in scored if s["overlap"] >= 0.15] or scored
    if night and not any(s["is_night"] for s in pool):
        print(f"WARNING: no night pass in {dt}; using the best daytime pass. Widen --days "
              f"(S1 passes Denmark ~05:30 and ~17:30 local; both are dark from about October to March).")
    pool.sort(key=lambda s: ((not s["is_night"]) if night else 0, s["ddays"], -round(s["overlap"], 2)))
    b = pool[0]
    print(f"{len(items)} candidate pass(es); chose {b['item'].id}")
    print(f"  {b['item'].datetime:%Y-%m-%d %H:%M} UTC, {b['orbit']}, sun {b['sun']:.0f} deg "
          f"({'night' if b['is_night'] else 'day/twilight'}), {b['overlap'] * 100:.0f}% of bbox covered")
    return b["item"]


def build_landmask(bounds_ll, crs, left, top, width, height, cell, out_path):
    """Land mask (1 = land) from ESA WorldCover on the grid (crs, left, top, cell) covering width x height metres.
    Reads only the needed window of each WorldCover tile. Areas with no WorldCover tile (open sea) count as water."""
    import planetary_computer as pc
    from pystac_client import Client
    from rasterio.transform import Affine
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    h, w = int(np.ceil(height / cell)), int(np.ceil(width / cell))
    dst_t = Affine(cell, 0, left, 0, -cell, top)
    land = np.zeros((h, w), np.uint8)
    cat = Client.open(PC_STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=[WORLDCOVER], bbox=list(bounds_ll)).items())
    for it in items:
        with rasterio.open(it.assets["map"].href) as src:
            pad = 0.01
            win = from_bounds(bounds_ll[0] - pad, bounds_ll[1] - pad, bounds_ll[2] + pad, bounds_ll[3] + pad,
                              transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                continue
            a = src.read(1, window=win)
            part = np.zeros((h, w), np.uint8)
            reproject(((a != 0) & (a != 80)).astype(np.uint8), part, src_transform=src.window_transform(win),
                      src_crs=src.crs, dst_transform=dst_t, dst_crs=crs, resampling=Resampling.mode)
            land |= part
    with rasterio.open(out_path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs=crs,
                       transform=dst_t, compress="deflate") as d:
        d.write(land, 1)
    print(f"  land mask: {len(items)} WorldCover tile(s), {100 * land.mean():.1f}% land -> {out_path}")
    return out_path


def assemble_s1(item, bbox, out_dir):
    """Read the bbox window of VV (+ VH) from the RTC COGs and write <out_dir>/<id>.S1 for pipeline_sar.py."""
    import json
    import sar as S
    tag = "_".join(f"{v:.2f}".replace("-", "m") for v in bbox)          # one pass can cover several areas
    folder = f"{out_dir}/{item.id}__{tag}.S1"
    os.makedirs(folder, exist_ok=True)
    sign = __import__("planetary_computer").sign
    geo = None
    for pol in ("vv", "vh"):
        if pol not in item.assets:
            continue
        with rasterio.open(sign(item.assets[pol].href)) as src:
            if geo is None:
                w = from_bounds(*transform_bounds("EPSG:4326", src.crs, *bbox), transform=src.transform)
                w = w.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
                if w.width < 16 or w.height < 16:
                    sys.exit("bbox does not overlap the chosen pass (empty window). Check lon/lat order.")
                geo = win_bounds(w, src.transform)
            else:
                w = from_bounds(*geo, transform=src.transform).round_offsets().round_lengths()
            a = src.read(1, window=w).astype(np.float32)
            a[~np.isfinite(a) | (a <= 0) | (a == (src.nodata if src.nodata is not None else -1))] = 0
            with rasterio.open(f"{folder}/{pol}.tif", "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1,
                               dtype="float32", crs=src.crs, transform=src.window_transform(w), nodata=0,
                               compress="deflate", tiled=True) as d:
                d.write(a, 1)
            crs, valid = src.crs, float((a > 0).mean())
    p = item.properties
    orbit = (p.get("sat:orbit_state") or "").lower() or None
    json.dump(dict(sensor="sentinel-1", id=item.id, time_utc=item.datetime.replace(tzinfo=None).isoformat(), pol="VV",
                   orbit_state=orbit, heading_deg=S.NOMINAL_HEADING.get(orbit), look_side="right",
                   incidence_deg=S.DEFAULT_INCIDENCE["sentinel-1"], resolution_m=S.DEFAULT_RESOLUTION["sentinel-1"],
                   mode=p.get("sar:instrument_mode"), platform=p.get("platform"),
                   product=f"{S1_COLLECTION} gamma0 (linear power)"), open(f"{folder}/scene.json", "w"), indent=2)
    left, bottom, right, top = geo
    build_landmask(bbox, crs, left, top, right - left, top - bottom, 20.0, f"{folder}/landmask.tif")
    return folder, valid


def landmask_for(path):
    """Write a WorldCover land mask next to an existing SAR scene (e.g. an ICEYE product)."""
    import sar as S
    sc = S.open_scene(path)
    out = (f"{path.rstrip('/')}/landmask.tif" if os.path.isdir(path)
           else f"{os.path.splitext(path)[0]}_landmask.tif")
    cell = max(20.0, sc.px)
    build_landmask(sc.lonlat_bounds(), sc.crs, sc.left, sc.top, sc.width * sc.px, sc.height * sc.px, cell, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", choices=["s2", "s1"], default="s2",
                    help="s2 = Sentinel-2 optical (default); s1 = Sentinel-1 radar (night + through cloud)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("minLon", "minLat", "maxLon", "maxLat"),
                    help="area of interest in lon/lat")
    ap.add_argument("--date", help="target date YYYY-MM-DD (nearest scene within +/- --days is picked)")
    ap.add_argument("--days", type=int, default=5, help="search window half-width in days (default 5)")
    ap.add_argument("--night", action="store_true", help="s1: prefer a pass with the sun more than 6 deg below the horizon")
    ap.add_argument("--max-cloud", type=float, default=30.0, help="s2: reject scenes cloudier than this %% (default 30)")
    ap.add_argument("--min-coverage", type=float, default=40.0,
                    help="s2: reject tiles with less than this %% real data (avoids swath-edge tiles; default 40)")
    ap.add_argument("--out", default=None, help="output directory (default s2_extracted / s1_extracted)")
    ap.add_argument("--landmask-for", metavar="SCENE",
                    help="only build a WorldCover land mask for an existing SAR scene (e.g. an ICEYE GeoTIFF) and exit")
    a = ap.parse_args()

    if a.landmask_for:
        return landmask_for(a.landmask_for)
    if not (a.bbox and a.date):
        ap.error("--bbox and --date are required (unless --landmask-for)")
    if a.sensor == "s1":
        item = search_s1(a.bbox, a.date, a.days, a.night)
        print("Reading bbox window of VV/VH from the Cloud-Optimized GeoTIFFs...")
        folder, valid = assemble_s1(item, a.bbox, a.out or "s1_extracted")
        print(f"\nWrote {folder}  ({valid * 100:.0f}% of the window inside the swath)")
        print(f"Next:\n  OUT=./out_s1 python pipeline_sar.py --scene {folder} --ais ais_window.csv --osm")
        return

    item = search_scene(a.bbox, a.date, a.days, a.max_cloud, a.min_coverage)
    sign = __import__("planetary_computer").sign
    assets = {b: sign(item.assets[b].href) for b in (*BANDS_10M, BAND_20M)}
    sensing = item.properties.get("datetime") or item.datetime.isoformat()
    print("Reading bbox windows from the Cloud-Optimized GeoTIFFs (range requests, not the whole tile)...")
    safe, valid_frac = assemble(assets, a.bbox, a.out or "s2_extracted", item.id, sensing)
    print(f"\nWrote {safe}  ({valid_frac*100:.0f}% of the fetched window is real data)")
    if valid_frac < 0.4:
        print("WARNING: most of the fetched window is nodata (swath edge). This tile barely covers your bbox -- "
              "expect few/no vessels and lots of edge clutter. Try a bbox nearer the tile centre, another --date, "
              "or raise --days so a better-covering pass can be chosen.")
    print(f"Next:\n  OUT=./out python pipeline_real.py --safe {safe} --ais ais_window.csv")


if __name__ == "__main__":
    main()
