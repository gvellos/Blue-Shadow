#!/usr/bin/env python3
"""Blue Shadow on SAR: Sentinel-1 or ICEYE radar + DMA AIS. Works at night and through cloud.

Usage:
  OUT=./out_s1    python pipeline_sar.py --scene s1_extracted/<id>.S1 --ais ais_window.csv [--osm]
  OUT=./out_iceye python pipeline_sar.py --scene path/to/ICEYE_..._GRD.tif --ais ais_window.csv
then  OUT=./out_s1 python build_ui.py

Same contract as pipeline_real.py (same reason codes, JSON outputs and UI), with what changes for radar:
  * scene      : sar.open_scene() -> linear backscatter on a metric grid; any pixel size (S1 10 m, ICEYE 0.5-3 m).
  * masks      : land from landmask.tif (ESA WorldCover, written by fetch_scene.py), 200 m coastal buffer, swath no-data.
                 No cloud mask: radar sees through cloud.
  * detector   : the same mask-aware two-threshold CFAR, with guard/background windows fixed in METRES so a 1 m ICEYE
                 pixel grid gets windows ten times wider in pixels than a 10 m Sentinel-1 grid.
  * Doppler    : a ship moving towards/away from the radar is displaced along the flight direction (~110 s x its
                 slant-range speed for S1: hundreds of metres). Each AIS position is moved by the predicted shift before
                 matching, so moving ships still match (--no-az-shift to compare).
  * limit      : "small below limit" = about one resolution cell (20 m for S1, ~4 m for ICEYE spot/strip), and the
                 detection-limit curve injects synthetic ship echoes into real clear-water patches of the scene.
"""
import argparse, glob, json, math, os

import numpy as np
from rasterio.warp import transform as rtransform
from rasterio.windows import Window
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.setdefault("OUT", "./out_sar")
import pipeline as P          # noqa: E402  (after OUT is set)
import pipeline_real as R     # noqa: E402
import sar as S               # noqa: E402

GUARD_M, BG_M = 410.0, 910.0                   # CFAR windows in metres (pipeline.py's 41 / 91 px at 10 m)
COAST_BUFFER_M = 200.0
INJECT_LENGTHS = [3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100, 150]
DEFAULT_ECHO_K = 0.03      # ship echo = k * sea backscatter * L^3 (m^3); equals the simulator's 0.03 L^2 at 10 m px for L = 100 m.
# L^3 rather than L^2: small boats (wood / GRP, low freeboard) return far less per m^2 than steel hulls, so scaling the
# echo of matched 90-230 m ships down with L^2 claimed ~100 % detection of 6 m boats on a real Kattegat scene.
# Target speckle false alarms per 100 km^2 for the automatic threshold. ICEYE is lower because its few-look speckle,
# once warped to UTM, has a heavier extreme tail than the fitted gamma predicts (~10x more spikes on the fixture);
# Sentinel-1 keeps 1 because a stricter threshold starts losing weak real echoes. Both measured with eval.py --sar.
DEFAULT_FA_RATE = {"sentinel-1": 1.0, "iceye": 0.1}
# Ship extent = pixels within CORE_DB of the ship's peak. A bright ship's point-spread side-lobes (the "cross" along the
# image lines) sit >= ~13 dB below its peak (sinc; ~20 dB with the usual weighting), so they fall outside the core.
CORE_DB = 13.0
# A detection is a side-lobe artifact of a brighter ship when it lies on one of that ship's image-line arms, within
# SIDELOBE_CELLS resolution cells, and is at least SIDELOBE_DB fainter.
SIDELOBE_CELLS, SIDELOBE_DB = 40, 10.0
RESOLUTION_M = 20.0        # set per scene by configure()
# Peak contrast (dB over the local sea) below which an unexplained detection is reported as LOW_CONTRAST_CLUTTER rather
# than DARK_CANDIDATE. On two real Sentinel-1 night scenes (Oresund, Kattegat) every AIS-confirmed ship was >= 12 dB,
# while the wind-texture false alarms of a rough sea clustered at ~7 dB (90 % below 9 dB). Still queued, low priority.
MIN_CONTRAST_DB = 10.0
P.PRIORITY.setdefault("LOW_CONTRAST_CLUTTER", 0.15)


def odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def configure(scene, kseed, klow):
    """Point the shared detector / explainer at this scene's pixel size and resolution."""
    global RESOLUTION_M
    RESOLUTION_M = scene.resolution_m
    R.PX = scene.px
    P.GUARD, P.BG = odd(GUARD_M / scene.px), odd(BG_M / scene.px)
    P.K_SEED, P.K_LOW = kseed, klow
    P.NOCOV_X = 1e18                          # DMA receivers cover the whole Danish EEZ
    P.BELOW_LIMIT_M = scene.below_limit_m
    if scene.resolution_m < 8:                # finer size bands where the sensor can actually resolve small boats
        P.BANDS = [("<8 m", 0, 8), ("8-15 m", 8, 15)] + P.BANDS[1:]


# --------------------------------------------------------------------- masks
def build_masks(scene):
    """(full-res exclusion mask, coarse distance-to-land in metres, coarse factor, stats)."""
    f = max(1, int(round(20.0 / scene.px)))
    cell = f * scene.px
    H, W = scene.height, scene.width
    land = scene.land_grid(cell)
    if land is None:
        print("WARNING: no land mask (landmask.tif). Coast will cause false alarms; build one with "
              "`python fetch_scene.py --landmask-for <scene>`.")
        land = np.zeros((math.ceil(H / f), math.ceil(W / f)), bool)
    d_land = (ndi.distance_transform_edt(~land, sampling=cell) if land.any()
              else np.full(land.shape, 1e5)).astype(np.float32)
    excl = np.repeat(np.repeat(d_land < COAST_BUFFER_M, f, 0), f, 1)[:H, :W]
    return excl, d_land, f, dict(land_pct=round(100 * float(land.mean()), 1), coarse_cell_m=cell,
                                 has_landmask=bool(scene.landmask))


# ----------------------------------------------------------------- threshold
def estimate_enl(scene, excl, rng, n=60, win=64, q=99.9):
    """(tail-matched ENL, std/mean) of open-water speckle. ENL: the gamma ENL whose 99.9th percentile (relative to
    the mean) equals the median 99.9th percentile over random water windows.

    Fitting the tail rather than mean^2/variance matters: resampling (warping to UTM) smooths the variance but keeps
    bright speckle spikes, and the spikes are what produce false alarms. Taking the median over windows ignores the
    odd window that holds a ship; fitting further out (99.99 %) was tried and is dominated by small boats.
    Low ENL (single-look, high resolution) = heavy tail = higher threshold."""
    from scipy.optimize import brentq
    from scipy.stats import gamma
    H, W = scene.height, scene.width
    win = min(win, H, W)
    tails, cvs = [], []
    for _ in range(n * 5):
        r, c = int(rng.integers(0, H - win + 1)), int(rng.integers(0, W - win + 1))
        if excl[r:r + win, c:c + win].any():
            continue
        img, valid = scene.read(Window(c, r, win, win))
        x = img[valid]
        if x.size < 0.9 * win * win or x.mean() <= 0:
            continue
        x = x / x.mean()
        tails.append(np.percentile(x, q))
        cvs.append(x.std())
        if len(tails) == n:
            break
    if not tails:
        return None, None
    target, p = float(np.median(tails)), 1 - q / 100
    f = lambda L: gamma.isf(p, L, scale=1.0 / L) - target
    lo, hi = 0.3, 100.0
    enl = lo if f(lo) < 0 else hi if f(hi) > 0 else float(brentq(f, lo, hi))
    return enl, float(np.median(cvs))


def auto_kseed(enl, cv, px, fa_per_100km2):
    """CFAR seed threshold giving `fa_per_100km2` expected speckle false alarms per 100 km^2 of water: the gamma(ENL)
    intensity level exceeded with that probability per pixel, expressed in the standard deviations the CFAR actually
    measures (cv = std / mean of water, which resampling makes smaller than 1/sqrt(ENL))."""
    from scipy.stats import gamma
    pfa = fa_per_100km2 * px * px / 1e8
    q = gamma.isf(pfa, enl, scale=1.0 / enl)
    return float(np.clip((q - 1.0) / cv, 6.0, 25.0))


# ----------------------------------------------------------------- detection
def detect(img, excl):
    """Two-pass CFAR. Pass 1 finds bright targets; pass 2 recomputes the background statistics with them (dilated)
    left out, so a ship in another ship's background ring no longer inflates the variance and hides it (the CFAR
    'masking effect' in convoys, anchorages and wind farms)."""
    z, mu = R.cfar(img, ~excl)
    bright = ndi.binary_dilation(z > P.K_SEED, iterations=3)
    if bright.any():
        z, mu = R.cfar(img, ~excl, bg_valid=~excl & ~bright)
    dets = R.blobs(img, z, mu, excl)
    for d in dets:
        measure_core(d, img, mu, z, excl)
    return dets


def measure_core(d, img, mu, z, excl):
    """Re-measure a detection's length on its core: the pixels of its detection blob (z > K_LOW) connected to the peak
    whose excess over the sea is within CORE_DB of the peak excess. The blob alone also swallows the side-lobe streaks
    of bright ships, which is why raw lengths came out 2-3x too long on real Sentinel-1; for a weak ship the blob is
    the tighter limit. The resolution blur is then removed (L^2 - res^2), since a point target already spreads over
    about one resolution cell."""
    px, r, c = R.PX, d["row"], d["col"]
    rad = int(min(d["L_est"] / px + 4, 600))
    r0, c0 = max(r - rad, 0), max(c - rad, 0)
    exc = img[r0:r + rad + 1, c0:c + rad + 1] - mu[r0:r + rad + 1, c0:c + rad + 1]
    peak = exc[r - r0, c - c0]
    if peak <= 0:
        return
    sl = (slice(r0, r + rad + 1), slice(c0, c + rad + 1))
    lab, _ = ndi.label((exc >= peak * 10 ** (-CORE_DB / 10)) & (z[sl] > P.K_LOW) & ~excl[sl], structure=np.ones((3, 3)))
    rr, cc = np.nonzero(lab == lab[r - r0, c - c0])
    lam = np.linalg.eigvalsh(np.cov(np.vstack([rr, cc]))).max() if len(rr) > 1 else 0.0
    raw = math.sqrt(12 * lam + 1) * px
    d["L_raw"], d["L_est"] = d["L_est"], float(math.sqrt(max(raw ** 2 - RESOLUTION_M ** 2, px ** 2)))
    d["excess_sum"], d["core_px"] = float(exc[rr, cc].sum()), int(len(rr))


def flag_sidelobes(dets, scene, protect=frozenset()):
    """Indices of detections that are side-lobe streaks of a brighter ship rather than targets of their own.

    A bright point target smears along the image's two axes: the flight (azimuth) direction and the range direction.
    A detection is flagged when a brighter detection exists such that (1) it lies within SIDELOBE_CELLS resolution
    cells, (2) its offset is (nearly) along the bright ship's azimuth or range axis, within half the bright ship's
    length plus two cells, and (3) it is at least SIDELOBE_DB fainter. Needs the flight direction."""
    if not scene.has_geometry or len(dets) < 2:
        return {}
    az, rg = scene.flight_unit(), scene.range_unit()
    reach, cell = SIDELOBE_CELLS * scene.resolution_m, scene.resolution_m
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["snr_db"])
    out = {}
    for k, i in enumerate(order):
        b = dets[i]
        if i in out:
            continue
        for j in order[k + 1:]:
            f = dets[j]
            if j in out or j in protect or f["snr_db"] > b["snr_db"] - SIDELOBE_DB:
                continue
            dxy = np.array([f["x"] - b["x"], f["y"] - b["y"]])
            dist = float(np.hypot(*dxy))
            if dist > reach:
                continue
            tol = 0.5 * b["L_est"] + 2 * cell
            if min(abs(float(dxy @ az)), abs(float(dxy @ rg))) <= tol:
                out[j] = dict(parent=i, dist_m=round(dist), db_below=round(b["snr_db"] - f["snr_db"], 1),
                              axis="azimuth" if abs(float(dxy @ rg)) <= tol else "range")
    return out


def detect_tiled(scene, excl):
    H, W, px, out = scene.height, scene.width, scene.px, []
    block, pad = max(2800, 3 * P.BG), P.BG
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            r1, c1 = min(r0 + block, H), min(c0 + block, W)
            ra, ca, rb, cb = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, H), min(c1 + pad, W)
            ex = excl[ra:rb, ca:cb]
            if ex.mean() > 0.99:
                continue
            img, valid = scene.read(Window(ca, ra, cb - ca, rb - ra))
            if not valid.any():
                continue
            for d in detect(img, ex | ~valid):
                x, y = d["x"] + ca * px, d["y"] + ra * px
                if c0 * px <= x < c1 * px and r0 * px <= y < r1 * px:     # keep only detections in the block core
                    d["x"], d["y"], d["row"], d["col"] = x, y, d["row"] + ra, d["col"] + ca
                    out.append(d)
            print(f"  block rows {r0}-{r1} cols {c0}-{c1}: {len(out)} detections so far")
    return out


# ----------------------------------------------------------------------- AIS
def ais_at_image(tracks, scene, use_shift):
    """Interpolate AIS to image time, then move each vessel to where the SAR will SHOW it (Doppler azimuth shift).
    The true position is kept in x_true / y_true; the gate grows by 15 % of the shift for geometry uncertainty."""
    ais = P.interpolate(tracks, 0.0)
    for v in ais:
        tr = tracks[v["mmsi"]]
        i = int(np.argmin(np.abs(tr[:, 1])))
        v["sog_kn"], v["cog_deg"] = float(tr[i, 4]), float(tr[i, 5])
        dx, dy = scene.azimuth_shift(v["sog_kn"], v["cog_deg"]) if use_shift else (0.0, 0.0)
        v["x_true"], v["y_true"] = v["x"], v["y"]
        v["x"], v["y"] = v["x"] + dx, v["y"] + dy
        v["az_shift_m"] = round(math.hypot(dx, dy), 1)
        v["radius"] += 0.15 * v["az_shift_m"]
    return ais


# ----------------------------------------------------------------- injection
def render_ship_sar(img, r, c, L, energy, px, rng, os_=4):
    """Add a ship echo: hull rectangle (L x L/6) rasterised at px/os_ and block-summed to pixel coverage, carrying
    `energy` (intensity x pixels) with 4-look speckle. Sub-cell boats spread over one cell -- the resolution limit."""
    wd = max(L / 6.0, 2.0)
    n = int(math.ceil(L / px / 2)) + 2
    k = (2 * n + 1) * os_
    g = (np.arange(k) + 0.5) / os_ - (n + 0.5)
    dc, dr = rng.uniform(-0.5, 0.5, 2)
    cc, rr = np.meshgrid(g - dc, g - dr)
    ang = rng.uniform(0, math.pi)
    u = (cc * math.cos(ang) + rr * math.sin(ang)) * px
    v = (-cc * math.sin(ang) + rr * math.cos(ang)) * px
    cov = ((np.abs(u) <= L / 2) & (np.abs(v) <= wd / 2)).reshape(2 * n + 1, os_, 2 * n + 1, os_).mean(axis=(1, 3))
    if r - n < 0 or c - n < 0 or r + n + 1 > img.shape[0] or c + n + 1 > img.shape[1] or cov.sum() == 0:
        return False
    img[r - n:r + n + 1, c - n:c + n + 1] += (energy * cov / cov.sum() * rng.gamma(4, 0.25, cov.shape)).astype(np.float32)
    return True


def pick_water_tiles(excl, d_land, f, px, k=3):
    H, W = excl.shape
    size = min(4096, max(1024, 3 * P.BG), H, W)
    if size < 2 * P.BG:
        return []
    cands, step = [], max(size // 2, 1)
    starts = lambda n: sorted(set(range(0, n - size + 1, step)) | {n - size})      # always include the far edge
    for r0 in starts(H):
        for c0 in starts(W):
            e = excl[r0:r0 + size, c0:c0 + size].mean()
            if e < 0.02 and d_land[r0 // f:(r0 + size) // f + 1, c0 // f:(c0 + size) // f + 1].min() > 1000:
                cands.append((e, r0, c0))
    cands.sort()
    out = []
    for _, r0, c0 in cands:
        if all(abs(r0 - a) + abs(c0 - b) >= size for a, b in out):
            out.append((r0, c0))
        if len(out) == k:
            break
    return [(r0, c0, size) for r0, c0 in out]


def injection_curve(scene, excl, tiles, rng, echo_k, lengths, n=40):
    px, base = scene.px, []
    for r0, c0, s in tiles:
        img, valid = scene.read(Window(c0, r0, s, s))
        ex = excl[r0:r0 + s, c0:c0 + s] | ~valid
        base.append((img, ex, detect(img, ex), float(np.median(img[~ex])) if (~ex).any() else 1.0))
    margin, spacing = int(160 / px) + 5, max(BG_M / 2 / px, 40)     # keep neighbours out of each other's guard ring
    res = []
    for L in lengths:
        hit = tot = fa = 0
        for tile, ex, dets0, sea in base:
            t, pts, tries, s = tile.copy(), [], 0, tile.shape[0]
            while len(pts) < n and tries < 5000:
                tries += 1
                r, c = int(rng.integers(margin, s - margin)), int(rng.integers(margin, s - margin))
                if ex[max(r - margin, 0):r + margin, max(c - margin, 0):c + margin].any():
                    continue
                if any(math.hypot(r - q[0], c - q[1]) < spacing for q in pts):
                    continue
                if any(math.hypot(r * px - d["y"], c * px - d["x"]) < 200 for d in dets0):
                    continue
                energy = echo_k * sea * L ** 3 * rng.lognormal(0, 0.4) / (px * px)
                if render_ship_sar(t, r, c, L, energy, px, rng):
                    pts.append((r, c))
            dets = detect(t, ex)
            near = lambda d, q: math.hypot(d["y"] - (q[0] + 0.5) * px, d["x"] - (q[1] + 0.5) * px) < 30 + L / 2
            # a hit = reported as a ship: CFAR fired AND it clears the contrast floor (else it is LOW_CONTRAST_CLUTTER)
            hit += sum(any(near(d, q) and d["snr_db"] >= MIN_CONTRAST_DB for d in dets) for q in pts)
            fa += sum(1 for d in dets if not any(near(d, q) for q in pts)
                      and not any(math.hypot(d["x"] - e["x"], d["y"] - e["y"]) < 60 for e in dets0))
            tot += len(pts)
        res.append(dict(length_m=L, p_det=round(hit / max(tot, 1), 3), n=tot,
                        false_alarms_per_tile=round(fa / max(len(base), 1), 2)))
    return res


def db_png(path, img, valid):
    db = 10 * np.log10(np.maximum(img, 1e-6))
    lo, hi = np.percentile(db[valid], [2, 99.8]) if valid.any() else (-25.0, 0.0)
    plt.imsave(path, np.clip((db - lo) / max(hi - lo, 0.1), 0, 1), cmap="gray")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="Blue Shadow SAR folder (<id>.S1) or ICEYE GeoTIFF / product folder")
    ap.add_argument("--ais", default="ais_window.csv")
    ap.add_argument("--kseed", type=float, default=None,
                    help="CFAR seed threshold in sigma; default = set from the measured speckle ENL and --fa-rate")
    ap.add_argument("--fa-rate", type=float, default=None,
                    help="target speckle false alarms per 100 km^2 of water for the automatic threshold "
                         f"(default {DEFAULT_FA_RATE['sentinel-1']:g} for Sentinel-1, {DEFAULT_FA_RATE['iceye']:g} for ICEYE)")
    ap.add_argument("--klow", type=float, default=5.0, help="CFAR grow threshold in sigma")
    ap.add_argument("--maxcrops", type=int, default=200)
    ap.add_argument("--structures", help="csv with lon,lat[,name] of turbines / buoys / platforms")
    ap.add_argument("--osm", action="store_true", help="fetch structures from OpenStreetMap (cached in structures_osm.csv)")
    ap.add_argument("--no-az-shift", action="store_true", help="match AIS at its true position (no Doppler correction)")
    ap.add_argument("--echo-k", type=float, default=None,
                    help="ship echo for injection: k x sea backscatter x L^3; default = median of matched vessels")
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--sensitive", help="lon0,lat0,lon1,lat1 corridor that boosts priority x1.5")
    ap.add_argument("--history", help="JSON file of prior-pass contacts (see pipeline_real.py / link_contacts.py)")
    g = ap.add_argument_group("metadata overrides (when the product's metadata is missing or wrong)")
    g.add_argument("--time", help="acquisition time UTC, e.g. 2025-11-20T04:31:12")
    g.add_argument("--heading", type=float, help="satellite flight direction, degrees clockwise from north")
    g.add_argument("--incidence", type=float, help="incidence angle at scene centre, degrees")
    g.add_argument("--calib", type=float, help="ICEYE calibration_factor (intensity = calib * DN^2)")
    g.add_argument("--landmask", help="GeoTIFF with 1 = land (any grid)")
    a = ap.parse_args()

    path = a.scene or (sorted(glob.glob("s1_extracted/*.S1")) or [None])[0]
    if not path:
        raise SystemExit("no --scene given and nothing in s1_extracted/ (run fetch_scene.py --sensor s1 first)")
    scene = S.open_scene(path, time=a.time, heading=a.heading, incidence=a.incidence, calib=a.calib, landmask=a.landmask)
    configure(scene, a.kseed or 10.0, a.klow)
    t_img, px, H, W = scene.time, scene.px, scene.height, scene.width
    left, top, crs = scene.left, scene.top, scene.crs
    os.makedirs(f"{OUT}/crops", exist_ok=True)
    lonlat = lambda xs, ys: rtransform(crs, "EPSG:4326", [left + x for x in xs], [top - y for y in ys])
    bounds_ll = scene.lonlat_bounds()
    lon_c, lat_c = scene.centre_lonlat()
    sun = round(S.sun_elevation_deg(lat_c, lon_c, t_img), 1)
    night = sun < -6
    print(scene.summary())
    if not scene.has_geometry and not a.no_az_shift:
        print("WARNING: no flight direction in metadata (pass --heading); Doppler shift correction disabled")

    # ---- masks + threshold
    excl, d_land, f, scene_stats = build_masks(scene)
    tiles = pick_water_tiles(excl, d_land, f, px)
    enl, cv = estimate_enl(scene, excl, np.random.default_rng(3))
    a.fa_rate = a.fa_rate if a.fa_rate is not None else DEFAULT_FA_RATE.get(scene.sensor, 1.0)
    if a.kseed is not None:
        k_src = "--kseed"
    elif enl:
        P.K_SEED = auto_kseed(enl, cv, px, a.fa_rate)
        k_src = f"gamma model: speckle ENL {enl:.1f} (tail fit), water std/mean {cv:.2f}, {a.fa_rate:g} false alarm / 100 km2"
    else:
        k_src = "default (no clear water to measure speckle)"
    print(f"CFAR: guard {P.GUARD} px / background {P.BG} px ({GUARD_M:.0f} / {BG_M:.0f} m), seed threshold "
          f"{P.K_SEED:.1f} sigma ({k_src}); small-boat limit {P.BELOW_LIMIT_M:.0f} m; sun elevation {sun} deg")
    at_c = lambda arr, x, y: arr[min(int(y / px) // f, arr.shape[0] - 1), min(int(x / px) // f, arr.shape[1] - 1)]
    at_f = lambda arr, x, y: arr[min(int(y / px), H - 1), min(int(x / px), W - 1)]
    dist_land = lambda x, y: float(at_c(d_land, x, y))

    # ---- AIS (+ Doppler shift) + structures
    tracks, known_len, static, meta, ais_stats = R.load_ais(a.ais, t_img, left, top, crs)
    use_shift = scene.has_geometry and not a.no_az_shift
    ais = ais_at_image(tracks, scene, use_shift)
    for v in ais:
        v["in_scene"] = 0 <= v["x"] < W * px and 0 <= v["y"] < H * px
        v["masked"] = v["in_scene"] and bool(at_f(excl, v["x"], v["y"]))
    ais_scene = [v for v in ais if v["in_scene"] and not v["masked"]]
    ais_masked = [v for v in ais if v["masked"]]
    shifts = [v["az_shift_m"] for v in ais_scene if v["fresh"]]
    med_shift = round(float(np.median(shifts)), 0) if shifts else 0.0
    print(f"AIS: {ais_stats['vessels']} vessels | in scene: {len(ais_scene)} | under land mask: {len(ais_masked)} | "
          f"fresh: {sum(v['fresh'] for v in ais_scene)} | Doppler shift {'on' if use_shift else 'off'}"
          + (f", median {med_shift:.0f} m, max {max(shifts):.0f} m" if use_shift and shifts else ""))
    for p in filter(None, [a.structures, R.fetch_osm_structures(bounds_ll) if a.osm else None]):
        static += R.load_structures(p, left, top, crs)
    static = [(xy, nm) for xy, nm in static if 0 <= xy[0] < W * px and 0 <= xy[1] < H * px]
    print(f"Structures in scene: {len(static)}")

    # ---- detection + matching
    print("Detecting (tiled, mask-aware CFAR)...")
    dets_all = detect_tiled(scene, excl)
    # A detection confirmed by a fresh AIS vessel is a real ship, even on a brighter ship's streak (a guard boat next to
    # a dredger): match first, then only unmatched detections can be side-lobe artifacts.
    art = flag_sidelobes(dets_all, scene, protect=set(P.match(dets_all, ais_scene)))
    keep = [i for i in range(len(dets_all)) if i not in art]
    dets = [dets_all[i] for i in keep]                           # artifacts take no further part
    make_id = lambda i_all: f"BS-{t_img:%Y%m%d}-{i_all + 1:04d}"
    print(f"Side-lobe artifacts: {len(art)} of {len(dets_all)} detections lie on a brighter ship's image-line streak")
    pairs = P.match(dets, ais_scene)
    matched_mmsi = {v["mmsi"] for v, _ in pairs.values()}

    # ---- detection limit: synthetic ship echoes in real water of this scene
    curve, echo_src, echo_k = [], "skipped", a.echo_k or DEFAULT_ECHO_K
    if not a.no_inject:
        ks = []
        for i, d in enumerate(dets):
            if i in pairs and pairs[i][0]["mmsi"] in known_len and 20 <= pairs[i][0]["length"] <= 150:
                img, valid = scene.read(Window(max(d["col"] - P.BG // 2, 0), max(d["row"] - P.BG // 2, 0), P.BG, P.BG))
                sea = float(np.median(img[valid])) if valid.any() else 0.0
                if sea > 0:
                    ks.append(d["excess_sum"] * px * px / (sea * pairs[i][0]["length"] ** 3))
        if a.echo_k:
            echo_src = "override"
        elif len(ks) >= 5:
            echo_k, echo_src = float(np.median(ks)), f"median of {len(ks)} matched 20-150 m vessels"
        else:
            echo_src = f"default (only {len(ks)} matched 20-150 m vessels)"
        lengths = [L for L in INJECT_LENGTHS if L >= scene.resolution_m / 4]
        print(f"Injection: {len(tiles)} clear-water tiles, ship echo k = {echo_k:.2f} ({echo_src})")
        curve = injection_curve(scene, excl, tiles, np.random.default_rng(11), echo_k, lengths) if tiles else []
    p_det = (lambda L: float(np.interp(L, [c["length_m"] for c in curve], [c["p_det"] for c in curve]))) if curve else None

    sens = None
    if a.sensitive:
        lo0, la0, lo1, la1 = map(float, a.sensitive.split(","))
        e, n = rtransform("EPSG:4326", crs, [lo0, lo1], [la0, la1])
        xs, ys = [x - left for x in e], [top - y for y in n]
        sens = (min(xs), min(ys), max(xs), max(ys))

    # ---- contacts
    sensor_name = "Sentinel-1" if scene.sensor == "sentinel-1" else "ICEYE"
    recs = []
    for i, d in enumerate(dets):
        dl = dist_land(d["x"], d["y"])
        conf, brk = P.confidence(d, dl)
        rec = dict(id=make_id(keep[i]), time_utc=t_img.isoformat() + "+00:00", sensor=scene.sensor,
                   confidence=conf, confidence_breakdown=brk, est_length_m=round(d["L_est"]),
                   est_length_uncertainty_m=round(scene.resolution_m / 2 + 0.3 * d["L_est"]),
                   snr_db=round(d["snr_db"], 1), peak_z=round(d["peak_z"], 1), area_px=d["area_px"],
                   map_xy=[round(d["x"], 1), round(d["y"], 1)])
        if i in pairs:
            v, dd = pairs[i]
            m = meta.get(v["mmsi"], {})
            rec.update(status="matched", reason_code="AIS_MATCH",
                       evidence=dict(mmsi=v["mmsi"], name=m.get("name"), ship_type=m.get("ship_type"),
                                     ais_length_m=v["length"] if v["mmsi"] in known_len else None, dist_m=round(dd),
                                     gate_m=round(v["radius"] + 0.5 * d["L_est"]), ais_method=v["method"],
                                     ais_age_s=round(v["age_s"]), doppler_shift_m=v["az_shift_m"]))
        else:
            code, text, ev = P.explain(d, ais_scene + ais_masked, matched_mmsi, [xy for xy, _ in static], dl)
            text = text.replace("Sentinel limit", f"{sensor_name} limit")
            if code in ("DARK_CANDIDATE", "SMALL_BELOW_LIMIT") and d["snr_db"] < MIN_CONTRAST_DB:
                code = "LOW_CONTRAST_CLUTTER"
                text = (f"Only {d['snr_db']:.1f} dB above the local sea (AIS-confirmed ships here are >= {MIN_CONTRAST_DB:.0f} dB): "
                        "most likely wind texture or a wave crest, possibly a small low-reflectivity boat.")
                ev = dict(contrast_db=round(d["snr_db"], 1), min_ship_contrast_db=MIN_CONTRAST_DB)
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
                       nearby_ais=[dict(mmsi=v["mmsi"], dist_m=round(math.hypot(d["x"] - v["x"], d["y"] - v["y"])),
                                        age_s=round(v["age_s"]), fresh=v["fresh"], doppler_shift_m=v["az_shift_m"],
                                        length_m=v["length"] if v["mmsi"] in known_len else None) for v in near])
        recs.append(rec)
    for j, info in sorted(art.items()):                          # kept for audit, never queued
        d, parent = dets_all[j], make_id(info["parent"])
        recs.append(dict(id=make_id(j), time_utc=t_img.isoformat() + "+00:00", sensor=scene.sensor, confidence=0.0,
                         est_length_m=round(d["L_est"]), snr_db=round(d["snr_db"], 1), peak_z=round(d["peak_z"], 1),
                         area_px=d["area_px"], map_xy=[round(d["x"], 1), round(d["y"], 1)], status="artifact",
                         reason_code="SIDELOBE_ARTIFACT",
                         reason_text=f"Side-lobe streak of {parent}: on its {info['axis']} line {info['dist_m']} m away, "
                                     f"{info['db_below']} dB fainter.",
                         evidence=dict(parent=parent, **{k: info[k] for k in ("axis", "dist_m", "db_below")})))
    lon, lat = lonlat([r["map_xy"][0] for r in recs], [r["map_xy"][1] for r in recs])
    for r, lo, la in zip(recs, lon, lat):
        r["lat"], r["lon"] = round(la, 5), round(lo, 5)

    # ---- cross-pass linking (same history file as the Sentinel-2 path, so S1 / S2 / ICEYE passes link together)
    links_proposed = 0
    if a.history:
        import linking as LK
        for r in recs:
            r["scene"] = scene.id
        hist = json.load(open(a.history)) if os.path.exists(a.history) else []
        for h in hist:
            h["t"] = LK.parse_time(h["time_utc"])
        points = LK.propose_links(recs, hist)
        links_proposed = sum(len(r.get("links", [])) for r in recs)
        json.dump([dict(id=p["id"], time_utc=p["t"].isoformat(), lat=p["lat"], lon=p["lon"], status=p.get("status"),
                        length=p.get("length"), mmsi=p.get("mmsi"), track_id=p.get("track_id"), scene=p.get("scene"))
                   for p in points], open(a.history, "w"), indent=2)
        print(f"Cross-pass: {links_proposed} links proposed to {len(hist)} prior contacts")

    queue = sorted([r for r in recs if r["status"] == "unidentified"], key=lambda r: -r["priority"])

    # ---- radar crops (~640 m across, dB) + overview
    cs = int(min(512, max(64, 640 / px), H, W))
    for r in queue[:a.maxcrops]:
        c0 = int(np.clip(r["map_xy"][0] / px - cs / 2, 0, W - cs))
        r0 = int(np.clip(r["map_xy"][1] / px - cs / 2, 0, H - cs))
        img, valid = scene.read(Window(c0, r0, cs, cs))
        db_png(f"{OUT}/crops/{r['id']}.png", img, valid)
        r["crop"] = f"crops/{r['id']}.png"
    sc = 1100 / max(H, W)
    ov_img, ov_valid = scene.read(out_shape=(max(int(H * sc), 1), max(int(W * sc), 1)))
    db_png(f"{OUT}/overview.jpg", ov_img, ov_valid)
    scene_stats["usable_pct"] = round(100 * float(ov_valid.mean() * (1 - excl[::max(1, int(1 / sc)), ::max(1, int(1 / sc))].mean())), 1)

    # ---- GeoJSON: detections + AIS at image time (true position; shift in properties)
    feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[r["lon"], r["lat"]]), properties=r) for r in recs]
    if ais_scene:
        lo, la = lonlat([v["x_true"] for v in ais_scene], [v["y_true"] for v in ais_scene])
        feats += [dict(type="Feature", geometry=dict(type="Point", coordinates=[x, y]),
                       properties=dict(kind="ais", mmsi=v["mmsi"], age_s=round(v["age_s"]), method=v["method"], fresh=v["fresh"],
                                       radius_m=round(v["radius"]), doppler_shift_m=v["az_shift_m"],
                                       length_m=v["length"] if v["mmsi"] in known_len else None,
                                       matched=v["mmsi"] in matched_mmsi, name=meta.get(v["mmsi"], {}).get("name")))
                  for v, x, y in zip(ais_scene, lo, la)]
    json.dump(dict(type="FeatureCollection", features=feats), open(f"{OUT}/contacts.geojson", "w"))

    # ---- size-band report
    rows = []
    for name, lo, hi in P.BANDS:
        ref = [v for v in ais_scene if v["fresh"] and v["mmsi"] in known_len and lo <= v["length"] < hi]
        det = [v for v in ref if v["mmsi"] in matched_mmsi]
        rows.append(dict(band=name, ais_vessels_in_scene=len(ref), ais_detected=len(det),
                         recall_vs_ais=round(len(det) / len(ref), 2) if ref else None, recall_95ci=R.wilson(len(det), len(ref)),
                         expected_detectable=round(sum(p_det(v["length"]) for v in ref), 1) if p_det else None,
                         ais_under_mask=sum(1 for v in ais_masked if v["mmsi"] in known_len and lo <= v["length"] < hi),
                         ais_stale=sum(1 for v in ais_scene if not v["fresh"] and v["mmsi"] in known_len and lo <= v["length"] < hi),
                         sim_truth_vessels=0, sim_truth_detected=0))
    reasons = {}
    for r in queue:
        reasons[r["reason_code"]] = reasons.get(r["reason_code"], 0) + 1
    report = dict(real=True, sensor=sensor_name, pol=scene.pol, scene=scene.id,
                  sensing_time_utc=t_img.isoformat() + "+00:00", night=night, sun_elevation_deg=sun,
                  note=f"REAL {sensor_name} SAR scene {t_img:%Y-%m-%d %H:%M} UTC ({'night' if night else 'daylight'} pass) + "
                       "DMA AIS. Radar images in darkness and through cloud. Moving ships appear displaced along the flight "
                       + ("direction; AIS positions were moved by the predicted Doppler shift before matching. " if use_shift
                          else "direction; Doppler shift correction was OFF. ")
                       + "Bright ships can leave azimuth 'ghosts' and wind farms side-lobes, both possible false alarms. "
                         "Recall uses AIS as reference (cooperative vessels only). The detection curve comes from synthetic "
                         "ship echoes pasted into real water of this scene.",
                  scene_stats=scene_stats, ais_stats=ais_stats, ais_in_visible_scene=len(ais_scene), ais_under_mask=len(ais_masked),
                  structures_in_scene=len(static), detections=len(dets), matched=len(pairs), unidentified=len(queue),
                  static_structures=sum(r["status"] == "static_structure" for r in recs), reason_breakdown=reasons,
                  sidelobe_artifacts=len(art),
                  size_bands=rows, detection_limit_curve=curve, below_limit_m=round(P.BELOW_LIMIT_M),
                  injection=dict(echo_k=round(echo_k, 2), hull_contrast_source=echo_src, tiles=tiles if not a.no_inject else []),
                  geometry=dict(pixel_m=round(px, 2), resolution_m=round(scene.resolution_m, 1), heading_deg=scene.heading_deg,
                                look_side=scene.look_side, incidence_deg=scene.incidence_deg, orbit_state=scene.orbit_state,
                                mode=scene.mode, az_shift_applied=use_shift, median_az_shift_m=med_shift,
                                cfar_guard_px=P.GUARD, cfar_background_px=P.BG),
                  map_size=[round(W * px / 10), round(H * px / 10)], kseed=round(P.K_SEED, 2), kseed_source=k_src,
                  speckle_enl=round(enl, 2) if enl else None, links_proposed=links_proposed)
    json.dump(queue, open(f"{OUT}/unidentified_contacts.json", "w"), indent=2)
    json.dump(recs, open(f"{OUT}/all_detections.json", "w"), indent=2)
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)

    # ---- figure: detection limit + size bands
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    if curve:
        a1.plot([c["length_m"] for c in curve], [c["p_det"] for c in curve], "o-")
        a1.set_xscale("log")
    a1.axvspan(1, P.BELOW_LIMIT_M, color="orange", alpha=.2)
    a1.text(1.2, .05, f"below {sensor_name}\nreliable limit", fontsize=9)
    a1.set(xlabel="vessel length (m)", ylabel="detection probability", ylim=(0, 1.05),
           title=f"{sensor_name} detection limit: synthetic echoes in real water")
    x = np.arange(len(rows))
    a2.bar(x - .3, [r["ais_vessels_in_scene"] for r in rows], .2, label="AIS vessels (fresh, visible)", color="#ccc")
    a2.bar(x - .1, [r["expected_detectable"] or 0 for r in rows], .2, label="expected detectable")
    a2.bar(x + .1, [r["ais_detected"] for r in rows], .2, label="detected (AIS-matched)")
    a2.bar(x + .3, [r["ais_under_mask"] for r in rows], .2, label="under land mask", color="#4af")
    a2.set_xticks(x); a2.set_xticklabels([r["band"] for r in rows]); a2.legend(); a2.set_title("AIS vessels in scene by size band")
    fig.savefig(f"{OUT}/size_band_and_limit.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    print(f"\nDetections {len(dets)} | matched {len(pairs)} | static {report['static_structures']} | "
          f"unidentified {len(queue)} | reasons {reasons}")
    print("band | AIS in scene | detected | recall | 95% CI | expected | under mask | stale")
    for r in rows:
        print(f"{r['band']:>8} | {r['ais_vessels_in_scene']:4d} | {r['ais_detected']:4d} | {r['recall_vs_ais']} | "
              f"{r['recall_95ci']} | {r['expected_detectable']} | {r['ais_under_mask']} | {r['ais_stale']}")
    if curve:
        print("detection curve:", [(c["length_m"], c["p_det"]) for c in curve])
    print(f"Outputs in {OUT}: unidentified_contacts.json, all_detections.json, report.json, contacts.geojson, overview.jpg, "
          "size_band_and_limit.png, crops/")


if __name__ == "__main__":
    main()
