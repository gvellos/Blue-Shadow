#!/usr/bin/env python3
"""Blue Shadow on REAL data: Sentinel-2 L2A (B08 NIR, 10 m) + DMA AIS (ais_window.csv from inspect_data.py).

Usage:  OUT=./out python pipeline_real.py [--safe s2_extracted/XXX.SAFE] [--ais ais_window.csv] [--kseed 12]
                                         [--structures structures.csv] [--osm] [--mask-cirrus]
                                         [--inject-pct 15] [--sensitive lon0,lat0,lon1,lat1]

Reuses interpolate / match / explain / confidence / BANDS from pipeline.py and swaps in real loaders:
  * scene      : B08 read block-wise (10980x10980 px never lives in memory at once); SCL 20 m gives land, cloud,
                 cirrus and no-data masks. Land/cloud blobs smaller than a ship-sized area are NOT masked, because
                 bright ships are often classified as cloud/unclassified by Sen2Cor.
  * detector   : same two-threshold CFAR, but background statistics ignore masked pixels (coast, cloud, no-data).
  * AIS        : clean (9-digit MMSI, duplicates, >60 kn jump outliers, SOG/COG sentinels), project through the
                 raster's UTM grid, pipeline.interpolate() to the tile sensing time from MTD_TL.xml.
  * structures : AtoN from AIS + optional csv (lon,lat[,name]) + optional OpenStreetMap fetch (--osm).
  * injection  : synthetic boats pasted into REAL clear-water patches of this scene with a sub-pixel coverage model,
                 giving a measured detection-probability-vs-length curve for this sea state and sun geometry.
Coordinates: local metres, x east / y south from the tile's top-left corner (same frame as pipeline.py)."""
import argparse, glob, json, math, os, re, urllib.parse, urllib.request
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as rtransform
from rasterio.windows import Window
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.setdefault("OUT", "./out")
import pipeline as P  # noqa: E402  (after OUT is set)

PX = 10.0
UNKNOWN_LEN = 30.0            # placeholder length for matching only; excluded from size-band reference
MIN_LAND_PX20 = 2000          # 0.8 km2 at 20 m: smaller "not water" blobs may be ships, not land
MIN_CLOUD_PX20 = 150          # 6 ha: bright ships (+wake) are often classed as cloud; keep small blobs
CLOUD_CLASSES = [3, 8, 9]     # cloud shadow, cloud medium / high probability (10 = thin cirrus, optional)
INJECT_LENGTHS = [6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100, 150]
REFL = lambda dn: np.maximum(dn.astype(np.float32) - 1000.0, 0) / 100.0   # DN -> % reflectance (baseline >= 04.00)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(c - h, 0), 2), round(min(c + h, 1), 2)]


# ------------------------------------------------------------------- Sentinel-2
def band_path(safe, res, name, required=True):
    hits = glob.glob(f"{safe}/GRANULE/*/IMG_DATA/R{res}m/*_{name}_{res}m.*")
    if not hits and required:
        raise SystemExit(f"missing band {name} at {res} m in {safe}")
    return hits[0] if hits else None


def sensing_time(safe):
    """Tile sensing time from MTD_TL.xml; the datatake start in the product name can be up to a minute earlier."""
    for f in glob.glob(f"{safe}/GRANULE/*/MTD_TL.xml"):
        m = re.search(r"<SENSING_TIME[^>]*>([^<]+)<", open(f).read())
        if m:
            t = pd.Timestamp(m.group(1))
            return t.tz_convert(None) if t.tzinfo else t
    return pd.Timestamp(re.search(r"_(\d{8}T\d{6})_", os.path.basename(safe)).group(1))


def remove_small(mask, min_px):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def load_masks(safe, H, W, mask_cirrus):
    with rasterio.open(band_path(safe, 20, "SCL")) as s:
        scl = s.read(1)
    cloud_cls = CLOUD_CLASSES + ([10] if mask_cirrus else [])
    land = remove_small(np.isin(scl, [1, 2, 4, 5, 7, 11]), MIN_LAND_PX20)
    cloud = ndi.binary_dilation(remove_small(np.isin(scl, cloud_cls), MIN_CLOUD_PX20), iterations=5)
    d_land20 = (ndi.distance_transform_edt(~land, sampling=20) if land.any() else np.full(scl.shape, 1e5)).astype(np.float32)
    excl20 = (d_land20 < 200) | cloud | (scl == 0)
    up = lambda a: np.repeat(np.repeat(a, 2, 0), 2, 1)[:H, :W]
    stats = dict(water_pct=round(100 * float((scl == 6).mean()), 1), land_pct=round(100 * float(land.mean()), 1),
                 cloud_masked_pct=round(100 * float(cloud.mean()), 1), cirrus_pct=round(100 * float((scl == 10).mean()), 1),
                 cirrus_masked=bool(mask_cirrus), usable_pct=round(100 * float(1 - excl20.mean()), 1))
    return up(excl20), up(scl == 10), d_land20, stats


# ---------------------------------------------------------------------- CFAR
def cfar(img, valid):
    """CFAR z-score and local background mean; masked (invalid) pixels are ignored in the statistics."""
    box = lambda a, k: ndi.uniform_filter(a, k, mode="reflect")
    nb, ng = P.BG ** 2, P.GUARD ** 2
    w = valid.astype(np.float32)
    aw = img * w
    n = box(w, P.BG) * nb - box(w, P.GUARD) * ng
    s1 = box(aw, P.BG) * nb - box(aw, P.GUARD) * ng
    s2 = box(aw * img, P.BG) * nb - box(aw * img, P.GUARD) * ng
    ok = n > 0.25 * (nb - ng)                                  # need enough unmasked background pixels
    n = np.maximum(n, 1)
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 1e-6)
    return np.where(ok & valid, (img - mu) / np.sqrt(var), 0).astype(np.float32), mu


def blobs(img, z, mu, excl):
    lab, _ = ndi.label((z > P.K_LOW) & ~excl, structure=np.ones((3, 3)))
    seeds = np.unique(lab[(z > P.K_SEED) & ~excl])
    seeds = seeds[seeds > 0]
    objs, dets = ndi.find_objects(lab), []
    for l in seeds:
        sl = objs[l - 1]
        rr, cc = np.nonzero(lab[sl] == l)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        w = np.maximum(img[rr, cc] - mu[rr, cc], 1e-3)       # excess over local water background
        lam = np.linalg.eigvalsh(np.cov(np.vstack([rr, cc]))).max() if len(rr) > 1 else 0.0
        pk = np.argmax(w)
        dets.append(dict(x=float(((cc + 0.5) * w).sum() / w.sum() * PX), y=float(((rr + 0.5) * w).sum() / w.sum() * PX),
                         L_est=float(math.sqrt(12 * lam + 1) * PX), area_px=int(len(rr)), peak_z=float(z[rr[pk], cc[pk]]),
                         peak_excess_pct=float(w[pk]), row=int(rr[pk]), col=int(cc[pk]),
                         snr_db=float(10 * math.log10(max(img[rr[pk], cc[pk]], 1e-3) / max(mu[rr[pk], cc[pk]], 1e-3)))))
    return dets


def detect(img, excl):
    z, mu = cfar(img, ~excl)
    return blobs(img, z, mu, excl)


def detect_tiled(src, excl, block=2800, pad=90):
    H, W, out = src.height, src.width, []
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            r1, c1 = min(r0 + block, H), min(c0 + block, W)
            ra, ca, rb, cb = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, H), min(c1 + pad, W)
            ex = excl[ra:rb, ca:cb]
            if ex.mean() > 0.99:
                continue
            dn = src.read(1, window=Window(ca, ra, cb - ca, rb - ra))
            for d in detect(REFL(dn), ex | (dn == 0)):
                x, y = d["x"] + ca * PX, d["y"] + ra * PX
                if c0 * PX <= x < c1 * PX and r0 * PX <= y < r1 * PX:   # keep only detections in the block core
                    d["x"], d["y"], d["row"], d["col"] = x, y, d["row"] + ra, d["col"] + ca
                    out.append(d)
            print(f"  block rows {r0}-{r1} cols {c0}-{c1}: {len(out)} detections so far")
    return out


# ----------------------------------------------------------------------- AIS
def load_ais(path, t_img, left, top, crs):
    """ais_window.csv -> (tracks in pipeline schema, MMSIs with known length, AtoN xy, per-MMSI metadata, stats)."""
    a = pd.read_csv(path, parse_dates=["t"], low_memory=False)
    stats = dict(messages_raw=int(len(a)))
    a = a.dropna(subset=["t", "MMSI", "Latitude", "Longitude"])
    a = a[a.MMSI.between(100_000_000, 999_999_999) & a.Latitude.between(-90, 90) & a.Longitude.between(-180, 180)]
    a = a.drop_duplicates(["MMSI", "t"])
    stats["type_of_mobile"] = a.drop_duplicates("MMSI")["Type of mobile"].value_counts().to_dict()
    atons = a[a["Type of mobile"] == "AtoN"].groupby("MMSI")[["Latitude", "Longitude"]].median()
    a = a[a["Type of mobile"].isin(["Class A", "Class B"])].copy()
    a["SOG"] = pd.to_numeric(a["SOG"], errors="coerce").where(lambda s: s.between(0, 60), 0.0).fillna(0.0)
    a["COG"] = pd.to_numeric(a["COG"], errors="coerce").where(lambda s: s.between(0, 359.9), 0.0).fillna(0.0)
    a["Length"] = pd.to_numeric(a["Length"], errors="coerce")
    known = a.groupby("MMSI")["Length"].max()
    known = set(known[known > 0].index)
    a["L"] = a["Length"].where(a["Length"] > 0, np.nan)
    a["L"] = a.groupby("MMSI")["L"].transform("max").fillna(UNKNOWN_LEN)
    e, n = rtransform("EPSG:4326", crs, a["Longitude"].tolist(), a["Latitude"].tolist())
    a["x"], a["y"] = np.array(e) - left, top - np.array(n)
    a["ts"] = (a["t"] - t_img).dt.total_seconds()
    a = a.sort_values(["MMSI", "ts"])
    # impossible jumps: a point implying > 60 kn both from its predecessor and to its successor is junk
    g = a.groupby("MMSI")
    v_p = np.hypot(g.x.diff(), g.y.diff()) / g.ts.diff().clip(lower=1)
    v_n = np.hypot(g.x.diff(-1), g.y.diff(-1)) / (-g.ts.diff(-1)).clip(lower=1)
    bad = ((v_p > 31) & (v_n > 31)).fillna(False)
    stats["messages_dropped_jump"], stats["messages_clean"] = int(bad.sum()), int((~bad).sum())
    a = a[~bad]
    tracks, meta = {}, {}
    for mmsi, g in a.groupby("MMSI"):
        tracks[int(mmsi)] = g[["MMSI", "ts", "x", "y", "SOG", "COG", "L"]].to_numpy(float)
        last = lambda c: (str(g[c].dropna().iloc[-1]) if c in g and g[c].notna().any() else None)
        meta[int(mmsi)] = dict(type_of_mobile=g["Type of mobile"].iloc[-1], name=last("Name"), ship_type=last("Ship type"),
                               n_msgs=int(len(g)))
    stats["vessels"] = len(tracks)
    ea, na = rtransform("EPSG:4326", crs, atons["Longitude"].tolist(), atons["Latitude"].tolist())
    static = [((float(x - left), float(top - y)), "AtoN (AIS)") for x, y in zip(ea, na)]
    return tracks, known, static, meta, stats


# ---------------------------------------------------------------- structures
def load_structures(path, left, top, crs):
    df = pd.read_csv(path)
    e, n = rtransform("EPSG:4326", crs, df.lon.tolist(), df.lat.tolist())
    names = df["name"].astype(str).tolist() if "name" in df else [os.path.basename(path)] * len(df)
    return [((float(x - left), float(top - y)), nm) for x, y, nm in zip(e, n, names)]


def fetch_osm_structures(lonlat_bounds, cache="structures_osm.csv"):
    """Offshore wind turbines, platforms, buoys, beacons and lights from OpenStreetMap (Overpass API), cached to csv."""
    if os.path.exists(cache):
        return cache
    lon0, lat0, lon1, lat1 = lonlat_bounds
    bbox = f"{lat0},{lon0},{lat1},{lon1}"
    q = f"""[out:json][timeout:120];(
      node["generator:source"="wind"]({bbox}); node["power"="generator"]["generator:method"="wind_turbine"]({bbox});
      node["man_made"="offshore_platform"]({bbox}); node["seamark:type"~"buoy|beacon|light|platform|mooring"]({bbox});
      node["man_made"="lighthouse"]({bbox}); way["man_made"="offshore_platform"]({bbox});
    );out center;"""
    print("Fetching structures from OpenStreetMap...")
    data = json.load(urllib.request.urlopen("https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(q), timeout=180))
    rows = []
    for e in data["elements"]:
        lat, lon = (e["lat"], e["lon"]) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
        t = e.get("tags", {})
        rows.append(dict(lon=lon, lat=lat, name=t.get("name") or t.get("seamark:type") or t.get("generator:source") or t.get("man_made") or "osm"))
    pd.DataFrame(rows, columns=["lon", "lat", "name"]).to_csv(cache, index=False)
    print(f"  {len(rows)} structures -> {cache}")
    return cache


# ----------------------------------------------------------------- injection
def render_boat_optical(img, r, c, L, rng, contrast_pct, os_=5):
    """Paste a boat into a NIR reflectance image: rotated rectangle rasterised at PX/os_ m and block-summed to pixel
    coverage. A 6 m boat covering 12 % of a 10 m pixel adds 12 % of the hull contrast; that is the small-boat limit."""
    wd = max(L / 5.0, 1.5)
    n = int(math.ceil(L / PX / 2)) + 2
    k = (2 * n + 1) * os_
    g = (np.arange(k) + 0.5) / os_ - (n + 0.5)
    dc, dr = rng.uniform(-0.5, 0.5, 2)
    cc, rr = np.meshgrid(g - dc, g - dr)
    ang = rng.uniform(0, math.pi)
    u = (cc * math.cos(ang) + rr * math.sin(ang)) * PX
    v = (-cc * math.sin(ang) + rr * math.cos(ang)) * PX
    cov = ((np.abs(u) <= L / 2) & (np.abs(v) <= wd / 2)).reshape(2 * n + 1, os_, 2 * n + 1, os_).mean(axis=(1, 3))
    if r - n < 0 or c - n < 0 or r + n + 1 > img.shape[0] or c + n + 1 > img.shape[1]:
        return False
    img[r - n:r + n + 1, c - n:c + n + 1] += (cov * contrast_pct * rng.lognormal(0, 0.15, cov.shape)).astype(np.float32)
    return True


def pick_water_tiles(excl, d_land20, k=3, size=1024):
    H, W = excl.shape
    cands = []
    for r0 in range(0, H - size, size // 2):
        for c0 in range(0, W - size, size // 2):
            e = excl[r0:r0 + size, c0:c0 + size].mean()
            if e < 0.02 and d_land20[r0 // 2:(r0 + size) // 2, c0 // 2:(c0 + size) // 2].min() > 1000:
                cands.append((e, r0, c0))
    cands.sort()
    out = []
    for _, r0, c0 in cands:                                      # spread picks over the scene
        if all(abs(r0 - a) + abs(c0 - b) >= 2 * size for a, b in out):
            out.append((r0, c0))
        if len(out) == k:
            break
    return [(r0, c0, size) for r0, c0 in out]


def injection_curve_real(src, excl, tiles, rng, contrast_pct, lengths=INJECT_LENGTHS, n=40):
    base = []
    for r0, c0, s in tiles:
        dn = src.read(1, window=Window(c0, r0, s, s))
        ex = excl[r0:r0 + s, c0:c0 + s] | (dn == 0)
        img = REFL(dn)
        base.append((img, ex, detect(img, ex)))
    res = []
    for L in lengths:
        hit = tot = fa = 0
        for tile, ex, dets0 in base:
            t, pts, tries, s = tile.copy(), [], 0, tile.shape[0]
            while len(pts) < n and tries < 5000:
                tries += 1
                r, c = int(rng.integers(40, s - 40)), int(rng.integers(40, s - 40))
                if ex[max(r - 30, 0):r + 30, max(c - 30, 0):c + 30].any() or any(math.hypot(r - q[0], c - q[1]) < 50 for q in pts):
                    continue
                if any(math.hypot(r * PX - d["y"], c * PX - d["x"]) < 200 for d in dets0):
                    continue
                if render_boat_optical(t, r, c, L, rng, contrast_pct):
                    pts.append((r, c))
            dets = detect(t, ex)
            near = lambda d, q: math.hypot(d["y"] - (q[0] + 0.5) * PX, d["x"] - (q[1] + 0.5) * PX) < 30 + L / 2
            hit += sum(any(near(d, q) for d in dets) for q in pts)
            fa += sum(1 for d in dets if not any(near(d, q) for q in pts)
                      and not any(math.hypot(d["x"] - e["x"], d["y"] - e["y"]) < 60 for e in dets0))
            tot += len(pts)
        res.append(dict(length_m=L, p_det=round(hit / max(tot, 1), 3), n=tot, false_alarms_per_tile=round(fa / max(len(base), 1), 2)))
    return res


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", default=None)
    ap.add_argument("--ais", default="ais_window.csv")
    ap.add_argument("--kseed", type=float, default=12.0, help="CFAR seed threshold in sigma (raise if too many false alarms)")
    ap.add_argument("--maxcrops", type=int, default=200)
    ap.add_argument("--structures", help="csv with lon,lat[,name] of turbines / buoys / platforms")
    ap.add_argument("--osm", action="store_true", help="fetch structures from OpenStreetMap (cached in structures_osm.csv)")
    ap.add_argument("--mask-cirrus", action="store_true", help="also exclude SCL thin-cirrus pixels (default: keep, flag, discount)")
    ap.add_argument("--inject-pct", type=float, default=None,
                    help="NIR reflectance (%%) excess of an injected hull over water; default = median of matched 20-100 m vessels")
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--sensitive", help="lon0,lat0,lon1,lat1 corridor that boosts priority x1.5")
    ap.add_argument("--history", help="JSON file of prior-pass contacts; cross-pass links are proposed against it "
                                      "and this pass is appended (see link_contacts.py for multi-run linking)")
    a = ap.parse_args()
    safe = (a.safe or glob.glob("s2_extracted/*.SAFE")[0]).rstrip("/")
    t_img = sensing_time(safe)
    P.K_SEED, P.NOCOV_X = a.kseed, 1e18      # DMA receivers cover the whole Danish EEZ: no "no coverage" strip
    os.makedirs(f"{OUT}/crops", exist_ok=True)

    src = rasterio.open(band_path(safe, 10, "B08"))
    H, W, left, top, crs = src.height, src.width, src.bounds.left, src.bounds.top, src.crs
    lonlat = lambda xs, ys: rtransform(crs, "EPSG:4326", [left + x for x in xs], [top - y for y in ys])
    lo_, la_ = lonlat([0, W * PX], [H * PX, 0])
    bounds_ll = (min(lo_), min(la_), max(lo_), max(la_))
    print(f"Scene {os.path.basename(safe)} {W}x{H} px, sensing time {t_img} UTC")

    # ---- masks
    excl, cirrus, d_land20, scene_stats = load_masks(safe, H, W, a.mask_cirrus)
    print(f"Masks: {scene_stats}")
    at20 = lambda arr, x, y: arr[min(int(y / PX) // 2, arr.shape[0] - 1), min(int(x / PX) // 2, arr.shape[1] - 1)]
    at10 = lambda arr, x, y: arr[min(int(y / PX), H - 1), min(int(x / PX), W - 1)]
    dist_land = lambda x, y: float(at20(d_land20, x, y))

    # ---- AIS + structures
    tracks, known_len, static, meta, ais_stats = load_ais(a.ais, t_img, left, top, crs)
    ais = P.interpolate(tracks, 0.0)
    for v in ais:
        v["in_scene"] = 0 <= v["x"] < W * PX and 0 <= v["y"] < H * PX
        v["masked"] = v["in_scene"] and bool(at10(excl, v["x"], v["y"]))
    ais_scene = [v for v in ais if v["in_scene"] and not v["masked"]]
    ais_masked = [v for v in ais if v["masked"]]
    print(f"AIS: {ais_stats['vessels']} vessels ({ais_stats['messages_dropped_jump']} jump outliers dropped) | in visible scene: "
          f"{len(ais_scene)} | under land/cloud mask: {len(ais_masked)} | fresh: {sum(v['fresh'] for v in ais_scene)}")
    for p in filter(None, [a.structures, fetch_osm_structures(bounds_ll) if a.osm else None]):
        static += load_structures(p, left, top, crs)
    static = [(xy, nm) for xy, nm in static if 0 <= xy[0] < W * PX and 0 <= xy[1] < H * PX]
    print(f"Structures in scene: {len(static)}")

    # ---- detection + matching
    print("Detecting (tiled, mask-aware CFAR)...")
    dets = detect_tiled(src, excl)
    pairs = P.match(dets, ais_scene)
    matched_mmsi = {v["mmsi"] for v, _ in pairs.values()}

    # ---- detection limit: synthetic boats in real water of this scene
    curve, tiles, contrast_src = [], [], "skipped"
    exc = [d["peak_excess_pct"] for i, d in enumerate(dets) if i in pairs and pairs[i][0]["mmsi"] in known_len and 20 <= pairs[i][0]["length"] <= 100]
    if a.inject_pct:
        contrast, contrast_src = a.inject_pct, "override"
    elif len(exc) >= 5:
        contrast, contrast_src = float(np.median(exc)), f"median of {len(exc)} matched 20-100 m vessels"
    else:
        contrast, contrast_src = 15.0, f"default (only {len(exc)} matched 20-100 m vessels)"
    if not a.no_inject:
        tiles = pick_water_tiles(excl, d_land20)
        print(f"Injection: {len(tiles)} clear-water tiles, hull contrast {contrast:.1f} % reflectance ({contrast_src})")
        curve = injection_curve_real(src, excl, tiles, np.random.default_rng(11), contrast) if tiles else []
    p_det = (lambda L: float(np.interp(L, [c["length_m"] for c in curve], [c["p_det"] for c in curve]))) if curve else None

    # ---- sensitive corridor (optional)
    sens = None
    if a.sensitive:
        lo0, la0, lo1, la1 = map(float, a.sensitive.split(","))
        e, n = rtransform("EPSG:4326", crs, [lo0, lo1], [la0, la1])
        xs, ys = [x - left for x in e], [top - y for y in n]
        sens = (min(xs), min(ys), max(xs), max(ys))

    # ---- contacts
    recs = []
    for i, d in enumerate(dets):
        dl = dist_land(d["x"], d["y"])
        conf, brk = P.confidence(d, dl)
        under_cirrus = bool(at10(cirrus, d["x"], d["y"]))
        if under_cirrus:
            brk["cirrus"] = 0.85
            conf = round(conf * 0.85, 3)
        rec = dict(id=f"BS-{t_img:%Y%m%d}-{i + 1:04d}", time_utc=t_img.isoformat() + "+00:00", confidence=conf,
                   confidence_breakdown=brk, est_length_m=round(d["L_est"]), est_length_uncertainty_m=10 + round(0.3 * d["L_est"]),
                   snr_db=round(d["snr_db"], 1), peak_z=round(d["peak_z"], 1), area_px=d["area_px"], under_cirrus=under_cirrus,
                   map_xy=[round(d["x"], 1), round(d["y"], 1)])
        if i in pairs:
            v, dd = pairs[i]
            m = meta.get(v["mmsi"], {})
            rec.update(status="matched", reason_code="AIS_MATCH",
                       evidence=dict(mmsi=v["mmsi"], name=m.get("name"), ship_type=m.get("ship_type"),
                                     ais_length_m=v["length"] if v["mmsi"] in known_len else None, dist_m=round(dd),
                                     gate_m=round(v["radius"] + 0.5 * d["L_est"]), ais_method=v["method"], ais_age_s=round(v["age_s"])))
        else:
            code, text, ev = P.explain(d, ais_scene + ais_masked, matched_mmsi, [xy for xy, _ in static], dl)
            if code == "STATIC_STRUCTURE":
                j = min(range(len(static)), key=lambda k: math.hypot(d["x"] - static[k][0][0], d["y"] - static[k][0][1]))
                ev = dict(structure=static[j][1], dist_m=round(math.hypot(d["x"] - static[j][0][0], d["y"] - static[j][0][1])))
                text = f"Within 150 m of catalogued structure ({static[j][1]})."
            if "mmsi" in ev and meta.get(ev["mmsi"], {}).get("name"):
                ev["name"] = meta[ev["mmsi"]]["name"]
            near = sorted(ais_scene, key=lambda v: math.hypot(d["x"] - v["x"], d["y"] - v["y"]))[:3]
            inz = bool(sens and sens[0] <= d["x"] <= sens[2] and sens[1] <= d["y"] <= sens[3])
            rec.update(status="static_structure" if code == "STATIC_STRUCTURE" else "unidentified", reason_code=code,
                       reason_text=text, evidence=ev, in_sensitive_zone=inz,
                       priority=round(P.PRIORITY[code] * conf * (1.5 if inz else 1), 3), links=[], analyst_state="new",
                       nearby_ais=[dict(mmsi=v["mmsi"], dist_m=round(math.hypot(d["x"] - v["x"], d["y"] - v["y"])), age_s=round(v["age_s"]),
                                        fresh=v["fresh"], length_m=v["length"] if v["mmsi"] in known_len else None) for v in near])
        recs.append(rec)
    lon, lat = lonlat([r["map_xy"][0] for r in recs], [r["map_xy"][1] for r in recs])
    for r, lo, la in zip(recs, lon, lat):
        r["lat"], r["lon"] = round(la, 5), round(lo, 5)

    # ---- cross-pass linking: propose tentative links to prior-pass contacts, accumulate a history file
    links_proposed = 0
    if a.history:
        import linking as LK
        for r in recs:
            r["scene"] = os.path.basename(safe)
        hist = json.load(open(a.history)) if os.path.exists(a.history) else []
        for h in hist:
            h["t"] = LK.parse_time(h["time_utc"])
        points = LK.propose_links(recs, hist)
        links_proposed = sum(len(r.get("links", [])) for r in recs)
        json.dump([dict(id=p["id"], time_utc=p["t"].isoformat(), lat=p["lat"], lon=p["lon"], status=p.get("status"),
                        length=p.get("length"), mmsi=p.get("mmsi"), track_id=p.get("track_id"), scene=p.get("scene"))
                   for p in points], open(a.history, "w"), indent=2)
        linked = sum(1 for r in recs if r.get("links"))
        print(f"Cross-pass: {links_proposed} links proposed to {len(hist)} prior contacts ({linked} of this pass's contacts linked)")

    queue = sorted([r for r in recs if r["status"] == "unidentified"], key=lambda r: -r["priority"])

    # ---- true-colour crops for the queue + overview
    bands = [rasterio.open(band_path(safe, 10, b)) for b in ("B04", "B03", "B02")]
    stretch = lambda x: np.clip(np.maximum(x.astype(np.float32) - 1000, 0) / 1200.0, 0, 1) ** (1 / 2.2)
    for r in queue[:a.maxcrops]:
        c0 = int(np.clip(r["map_xy"][0] / PX - 32, 0, W - 64)); r0 = int(np.clip(r["map_xy"][1] / PX - 32, 0, H - 64))
        rgb = np.dstack([stretch(b.read(1, window=Window(c0, r0, 64, 64))) for b in bands])
        plt.imsave(f"{OUT}/crops/{r['id']}.png", rgb)
        r["crop"] = f"crops/{r['id']}.png"
        nir = REFL(src.read(1, window=Window(c0, r0, 64, 64)))
        lo_v, hi_v = np.percentile(nir, [2, 99.8])
        plt.imsave(f"{OUT}/crops/{r['id']}_nir.png", np.clip((nir - lo_v) / max(hi_v - lo_v, 0.1), 0, 1), cmap="gray")
        r["crop_nir"] = f"crops/{r['id']}_nir.png"
    ov = np.dstack([stretch(b.read(1, out_shape=(1098, 1098), resampling=Resampling.average)) for b in bands])
    plt.imsave(f"{OUT}/overview.jpg", ov)
    for b in bands:
        b.close()

    # ---- GeoJSON for any map client: detections + interpolated AIS positions
    feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[r["lon"], r["lat"]]), properties=r) for r in recs]
    if ais_scene:
        lo, la = lonlat([v["x"] for v in ais_scene], [v["y"] for v in ais_scene])
        feats += [dict(type="Feature", geometry=dict(type="Point", coordinates=[x, y]),
                       properties=dict(kind="ais", mmsi=v["mmsi"], age_s=round(v["age_s"]), method=v["method"], fresh=v["fresh"],
                                       radius_m=round(v["radius"]), length_m=v["length"] if v["mmsi"] in known_len else None,
                                       matched=v["mmsi"] in matched_mmsi, name=meta.get(v["mmsi"], {}).get("name")))
                  for v, x, y in zip(ais_scene, lo, la)]
    json.dump(dict(type="FeatureCollection", features=feats), open(f"{OUT}/contacts.geojson", "w"))

    # ---- size-band report: measured recall vs AIS length (known lengths, visible area only)
    rows = []
    for name, lo, hi in P.BANDS:
        ref = [v for v in ais_scene if v["fresh"] and v["mmsi"] in known_len and lo <= v["length"] < hi]
        det = [v for v in ref if v["mmsi"] in matched_mmsi]
        rows.append(dict(band=name, ais_vessels_in_scene=len(ref), ais_detected=len(det),
                         recall_vs_ais=round(len(det) / len(ref), 2) if ref else None, recall_95ci=wilson(len(det), len(ref)),
                         expected_detectable=round(sum(p_det(v["length"]) for v in ref), 1) if p_det else None,
                         ais_under_mask=sum(1 for v in ais_masked if v["mmsi"] in known_len and lo <= v["length"] < hi),
                         ais_stale=sum(1 for v in ais_scene if not v["fresh"] and v["mmsi"] in known_len and lo <= v["length"] < hi),
                         sim_truth_vessels=0, sim_truth_detected=0))
    reasons = {}
    for r in queue:
        reasons[r["reason_code"]] = reasons.get(r["reason_code"], 0) + 1
    report = dict(real=True, scene=os.path.basename(safe), sensing_time_utc=t_img.isoformat() + "+00:00",
                  note=f"REAL Sentinel-2 L2A scene {t_img:%Y-%m-%d %H:%M} UTC + DMA AIS. Optical NIR, not SAR: detection depends on "
                       "sun and cloud, wakes inflate length estimates. Recall uses AIS as reference (cooperative vessels only; "
                       "vessels in masked coastal/cloud areas are listed, not counted). The detection curve comes from synthetic "
                       "boats pasted into real water of this scene; real small boats vary in material and colour.",
                  scene_stats=scene_stats, ais_stats=ais_stats, ais_in_visible_scene=len(ais_scene), ais_under_mask=len(ais_masked),
                  structures_in_scene=len(static), detections=len(dets), matched=len(pairs), unidentified=len(queue),
                  static_structures=sum(r["status"] == "static_structure" for r in recs), reason_breakdown=reasons,
                  size_bands=rows, detection_limit_curve=curve,
                  injection=dict(hull_contrast_pct=round(contrast, 1), hull_contrast_source=contrast_src, tiles=tiles),
                  map_size=[W, H], kseed=a.kseed, links_proposed=links_proposed)
    json.dump(queue, open(f"{OUT}/unidentified_contacts.json", "w"), indent=2)
    json.dump(recs, open(f"{OUT}/all_detections.json", "w"), indent=2)
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)

    # ---- figure: detection limit + size bands
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    if curve:
        a1.plot([c["length_m"] for c in curve], [c["p_det"] for c in curve], "o-")
        a1.set_xscale("log")
    a1.axvspan(1, P.BELOW_LIMIT_M, color="orange", alpha=.2); a1.text(6, .05, "below Sentinel\nreliable limit", fontsize=9)
    a1.set(xlabel="vessel length (m)", ylabel="detection probability", ylim=(0, 1.05),
           title=f"Detection limit: synthetic boats in real water ({contrast:.0f}% hull contrast)")
    x = np.arange(len(rows))
    a2.bar(x - .3, [r["ais_vessels_in_scene"] for r in rows], .2, label="AIS vessels (fresh, visible)", color="#ccc")
    a2.bar(x - .1, [r["expected_detectable"] or 0 for r in rows], .2, label="expected detectable")
    a2.bar(x + .1, [r["ais_detected"] for r in rows], .2, label="detected (AIS-matched)")
    a2.bar(x + .3, [r["ais_under_mask"] for r in rows], .2, label="under land/cloud mask", color="#4af")
    a2.set_xticks(x); a2.set_xticklabels([r["band"] for r in rows]); a2.legend(); a2.set_title("AIS vessels in scene by size band")
    fig.savefig(f"{OUT}/size_band_and_limit.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    src.close()

    print(f"\nDetections {len(dets)} | matched {len(pairs)} | static {report['static_structures']} | unidentified {len(queue)} | reasons {reasons}")
    print("band | AIS in scene | detected | recall | 95% CI | expected | under mask | stale")
    for r in rows:
        print(f"{r['band']:>8} | {r['ais_vessels_in_scene']:4d} | {r['ais_detected']:4d} | {r['recall_vs_ais']} | {r['recall_95ci']} | "
              f"{r['expected_detectable']} | {r['ais_under_mask']} | {r['ais_stale']}")
    if curve:
        print("detection curve:", [(c["length_m"], c["p_det"]) for c in curve])
    print(f"Outputs in {OUT}: unidentified_contacts.json, all_detections.json, report.json, contacts.geojson, overview.jpg, "
          "size_band_and_limit.png, crops/")


if __name__ == "__main__":
    main()
