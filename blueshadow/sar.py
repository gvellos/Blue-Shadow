"""SAR scene loading for Blue Shadow: one interface for Sentinel-1 and ICEYE.

SAR sends its own microwave pulse, so it images by day, by night and through cloud. That is what closes the night gap
of the Sentinel-2 path: Sentinel-1 passes Denmark around 05:30 and 17:30 local time, ICEYE can be tasked at any hour.

Every loader returns a SarScene on a north-up metric grid with square pixels (warped on the fly when the product is in
radar geometry or lon/lat), so pipeline_sar.py can use the same frame as the rest of Blue Shadow:
local metres, x east / y south from the grid's top-left corner.

Supported inputs
  * Blue Shadow SAR folder  <id>.S1/  written by `python -m blueshadow fetch --sensor s1` (or make_fake_sar.py):
        vv.tif [vh.tif]   linear backscatter (Planetary Computer sentinel-1-rtc, gamma0, UTM, 10 m)
        landmask.tif      1 = land (ESA WorldCover), any grid; optional
        scene.json        sensor, id, time_utc, orbit_state, heading_deg, incidence_deg, resolution_m, ...
  * ICEYE GRD product: a GeoTIFF (amplitude) plus its .json or .xml metadata sidecar, or a folder holding them.
    Georeferencing may be a map grid, lon/lat or GCPs; all are warped to UTM. Intensity = calibration_factor * DN^2.
    The metadata key names follow ICEYE's product format guide; anything missing can be overridden (see open_scene).
    Not yet tested on a real ICEYE delivery -- check the printed scene summary against the product's metadata.
"""
import glob, json, math, os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.transform
from rasterio.transform import Affine, from_gcps
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, transform as rtransform

V_SAT = 7600.0                        # m/s, orbital speed (S1 and ICEYE are both ~7.5-7.6 km/s)
ALTITUDE_M = {"sentinel-1": 693e3, "iceye": 560e3}
NOMINAL_HEADING = {"ascending": 348.0, "descending": 192.0}   # flight direction at ~55 N, degrees from north
DEFAULT_INCIDENCE = {"sentinel-1": 37.0, "iceye": 25.0}
DEFAULT_RESOLUTION = {"sentinel-1": 20.0}                       # IW GRD-HD: ~20 m resolution on a 10 m grid


class SarScene:
    """A SAR image on a north-up metric grid plus the acquisition geometry needed to predict where moving ships appear."""

    def __init__(self, path, sensor, src, time, *, scene_id=None, scale=1.0, amplitude=False, pol="VV",
                 heading_deg=None, orbit_state=None, look_side="right", incidence_deg=None, altitude_m=None,
                 resolution_m=None, mode=None, landmask=None, meta=None):
        self.path, self.sensor, self.src, self.time = path, sensor, src, time
        self.id = scene_id or os.path.basename(os.path.normpath(path))
        self.scale, self.amplitude, self.pol = float(scale), bool(amplitude), pol
        self.orbit_state = (orbit_state or "").lower() or None
        self.heading_deg = heading_deg if heading_deg is not None else NOMINAL_HEADING.get(self.orbit_state)
        self.look_side = (look_side or "right").lower()
        self.incidence_deg = float(incidence_deg or DEFAULT_INCIDENCE.get(sensor, 35.0))
        self.altitude_m = float(altitude_m or ALTITUDE_M.get(sensor, 650e3))
        self.crs, self.transform = src.crs, src.transform
        self.width, self.height = src.width, src.height
        self.px = float(abs(src.transform.a))
        self.left, self.top = float(src.transform.c), float(src.transform.f)
        self.resolution_m = float(resolution_m or DEFAULT_RESOLUTION.get(sensor, self.px))
        self.mode, self.landmask, self.meta = mode, landmask, meta or {}

    # ------------------------------------------------------------------ pixels
    def read(self, window=None, out_shape=None):
        """Linear backscatter intensity (float32) and a validity mask (False = no data / outside the swath)."""
        kw = dict(window=window)
        if out_shape is not None:
            kw.update(out_shape=out_shape, resampling=Resampling.average)
        a = self.src.read(1, **kw).astype(np.float32)
        valid = np.isfinite(a) & (a > 0)
        if self.src.nodata is not None and np.isfinite(self.src.nodata):
            valid &= a != self.src.nodata
        a = np.where(valid, a, 0)
        img = self.scale * (a * a if self.amplitude else a)
        return img.astype(np.float32), valid

    @property
    def below_limit_m(self):
        """Length at or below which a detection can't be told from clutter: about one resolution cell, never under 4 m."""
        return max(self.resolution_m, 4.0)

    # ---------------------------------------------------------------- geometry
    @property
    def has_geometry(self):
        return self.heading_deg is not None

    def flight_unit(self):
        h = math.radians(self.heading_deg)
        return np.array([math.sin(h), -math.cos(h)])            # (east, south)

    def range_unit(self):
        """Ground direction pointing away from the satellite (right- or left-looking)."""
        h = math.radians(self.heading_deg + (90.0 if self.look_side == "right" else -90.0))
        return np.array([math.sin(h), -math.cos(h)])

    def azimuth_shift(self, sog_kn, cog_deg):
        """Displacement (dx east, dy south, metres) between where a moving ship IS and where SAR SHOWS it.

        A ship moving towards or away from the radar adds a Doppler shift, so the processor puts it in the wrong place
        along the flight direction: shift = -(R / V) * v_r, with R the slant range, V the orbital speed and v_r the
        slant-range speed (+ = moving away). At Sentinel-1 geometry R/V is about 110 s, so a 10 kn ship crossing the
        swath lands ~300 m off its AIS position. Flat-earth approximation, good to ~10 %."""
        if not self.has_geometry or not sog_kn:
            return np.zeros(2)
        c = math.radians(cog_deg)
        v = sog_kn * 0.5144 * np.array([math.sin(c), -math.cos(c)])
        inc = math.radians(self.incidence_deg)
        v_r = float(v @ self.range_unit()) * math.sin(inc)
        slant = self.altitude_m / math.cos(inc)
        return -(slant / V_SAT) * v_r * self.flight_unit()

    # ---------------------------------------------------------------- land mask
    def land_grid(self, cell_m):
        """Land mask resampled to a coarse grid aligned with the scene (cell_m must be a multiple of px).
        Returns None when the scene has no land mask."""
        if not self.landmask:
            return None
        f = max(1, int(round(cell_m / self.px)))
        h, w = math.ceil(self.height / f), math.ceil(self.width / f)
        t = Affine(self.px * f, 0, self.left, 0, -self.px * f, self.top)
        out = np.zeros((h, w), np.uint8)
        with rasterio.open(self.landmask) as lm:
            reproject(lm.read(1), out, src_transform=lm.transform, src_crs=lm.crs, dst_transform=t, dst_crs=self.crs,
                      resampling=Resampling.nearest)
        return out > 0

    def lonlat_bounds(self):
        xs = [self.left, self.left + self.width * self.px]
        ys = [self.top - self.height * self.px, self.top]
        lo, la = rtransform(self.crs, "EPSG:4326", [xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return min(lo), min(la), max(lo), max(la)

    def summary(self):
        night = sun_elevation_deg(*self.centre_lonlat()[::-1], self.time) < -6
        return (f"{self.sensor} {self.id}: {self.width}x{self.height} px at {self.px:.2f} m "
                f"(resolution ~{self.resolution_m:.1f} m), {self.pol}, {self.time:%Y-%m-%d %H:%M:%S} UTC "
                f"({'night' if night else 'day/twilight'}), orbit {self.orbit_state or '?'}, heading "
                f"{'?' if self.heading_deg is None else f'{self.heading_deg:.0f}'} deg, {self.look_side}-looking, "
                f"incidence {self.incidence_deg:.0f} deg, land mask {'yes' if self.landmask else 'NO'}")

    def centre_lonlat(self):
        lo0, la0, lo1, la1 = self.lonlat_bounds()
        return (lo0 + lo1) / 2, (la0 + la1) / 2


# ------------------------------------------------------------------ helpers
def sun_elevation_deg(lat, lon, t):
    """Approximate solar elevation (NOAA formulas, ~0.5 deg). t is a naive UTC timestamp."""
    t = pd.Timestamp(t)
    doy = t.dayofyear
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    ha = math.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    s = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def utm_for(lon, lat):
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def _to_utc(s):
    t = pd.Timestamp(s)
    return t.tz_convert(None) if t.tzinfo else t


def metric_grid(src):
    """`src` itself if it is already a north-up projected grid with square pixels, else a WarpedVRT onto UTM.
    Handles radar-geometry products (GCPs) and lon/lat products."""
    t, (gcps, gcrs) = src.transform, src.gcps
    if src.crs and src.crs.is_projected and t.b == 0 and t.d == 0 and abs(abs(t.a) - abs(t.e)) < 1e-6 * abs(t.a):
        return src
    if gcps:
        # Affine fit to the GCPs (a ground-range product over a few tens of km is close to affine); warn if it isn't.
        st = from_gcps(gcps)
        err = max(math.hypot(*(np.array(st * (g.col, g.row)) - (g.x, g.y))) for g in gcps)
        lo, la = rtransform(gcrs, "EPSG:4326", [g.x for g in gcps], [g.y for g in gcps])
        dst = utm_for(float(np.mean(lo)), float(np.mean(la)))
        unit_m = 111320.0 if gcrs.is_geographic else 1.0
        pix_m = math.hypot(st.a, st.d) * unit_m * (math.cos(math.radians(float(np.mean(la)))) if gcrs.is_geographic else 1)
        if err * unit_m > 2 * pix_m:
            print(f"WARNING: GCPs deviate up to {err * unit_m:.0f} m from an affine fit; positions may be off by that much")
        b = rasterio.transform.array_bounds(src.height, src.width, st)
        tr, w, h = calculate_default_transform(gcrs, dst, src.width, src.height, *b)       # b = (west, south, east, north)
        return WarpedVRT(src, src_crs=gcrs, src_transform=st, crs=dst, transform=tr, width=w, height=h,
                         resampling=Resampling.bilinear, nodata=0)
    if src.crs is None:
        raise SystemExit(f"{src.name}: no georeferencing (no CRS and no GCPs)")
    b = src.bounds
    lo, la = rtransform(src.crs, "EPSG:4326", [(b.left + b.right) / 2], [(b.top + b.bottom) / 2])
    dst = utm_for(lo[0], la[0])
    tr, w, h = calculate_default_transform(src.crs, dst, src.width, src.height, *b)
    return WarpedVRT(src, crs=dst, transform=tr, width=w, height=h, resampling=Resampling.bilinear, nodata=0)


def _flatten_json(obj, out, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_json(v, out, f"{prefix}{k}.".lower())
    elif isinstance(obj, list):
        for v in obj[:1]:                        # first element is enough for scalar metadata
            _flatten_json(v, out, prefix)
    else:
        out.setdefault(prefix.rstrip(".").split(".")[-1], obj)


def read_iceye_meta(tif):
    """Flat {lowercase key: value} from the ICEYE .json / .xml sidecar (first occurrence of each key wins),
    falling back to the GeoTIFF's own tags."""
    stem, folder = os.path.splitext(tif)[0], os.path.dirname(tif) or "."
    cands = [stem + ".json", stem + ".xml"] + sorted(glob.glob(f"{folder}/*.json")) + sorted(glob.glob(f"{folder}/*.xml"))
    meta = {}
    for p in cands:
        if not os.path.isfile(p) or p.endswith("scene.json"):
            continue
        if p.endswith(".json"):
            _flatten_json(json.load(open(p)), meta)
        else:
            for el in ET.parse(p).iter():
                tag = el.tag.split("}")[-1].lower()
                if el.text and el.text.strip():
                    meta.setdefault(tag, el.text.strip())
        meta["_source"] = p
        break
    with rasterio.open(tif) as s:
        for k, v in s.tags().items():
            meta.setdefault(k.lower(), v)
    return meta


def _pick(meta, *keys, cast=float):
    for k in keys:
        if k in meta and meta[k] not in (None, ""):
            try:
                return cast(meta[k])
            except (TypeError, ValueError):
                pass
    return None


def _find_landmask(path):
    stem = os.path.splitext(path)[0]
    for p in (f"{path}/landmask.tif", f"{stem}_landmask.tif", f"{os.path.dirname(path) or '.'}/landmask.tif"):
        if os.path.isfile(p):
            return p
    return None


# ------------------------------------------------------------------ loaders
def open_blueshadow_dir(path, **ov):
    m = json.load(open(f"{path}/scene.json"))
    pol = ov.get("pol") or m.get("pol", "VV")
    tif = f"{path}/{pol.lower()}.tif"
    if not os.path.isfile(tif):
        tif = sorted(glob.glob(f"{path}/v*.tif"))[0]
    src = metric_grid(rasterio.open(tif))
    return SarScene(path, m.get("sensor", "sentinel-1"), src, _to_utc(ov.get("time") or m["time_utc"]),
                    scene_id=m.get("id"), scale=m.get("scale", 1.0), amplitude=m.get("amplitude", False), pol=pol,
                    heading_deg=ov.get("heading") if ov.get("heading") is not None else m.get("heading_deg"),
                    orbit_state=m.get("orbit_state"), look_side=m.get("look_side", "right"),
                    incidence_deg=ov.get("incidence") or m.get("incidence_deg"), altitude_m=m.get("altitude_m"),
                    resolution_m=m.get("resolution_m"), mode=m.get("mode"),
                    landmask=ov.get("landmask") or _find_landmask(path), meta=m)


def open_iceye(path, **ov):
    if os.path.isdir(path):
        tifs = [p for p in sorted(glob.glob(f"{path}/*.tif")) if "landmask" not in os.path.basename(p)]
        if not tifs:
            raise SystemExit(f"no GeoTIFF in {path}")
        tif = tifs[0]
    else:
        tif = path
    meta = read_iceye_meta(tif)
    t0 = _pick(meta, "acquisition_start_utc", "zerodoppler_start_utc", "first_line_time", cast=_to_utc)
    t1 = _pick(meta, "acquisition_end_utc", "zerodoppler_end_utc", "last_line_time", cast=_to_utc)
    t = _to_utc(ov["time"]) if ov.get("time") else (t0 + (t1 - t0) / 2 if t0 is not None and t1 is not None else t0)
    if t is None:
        raise SystemExit(f"{tif}: acquisition time not found in metadata; pass --time YYYY-MM-DDTHH:MM:SS")
    calib = ov.get("calib") or _pick(meta, "calibration_factor", "calibration_constant")
    if calib is None:
        print("WARNING: no calibration_factor in ICEYE metadata; using 1.0 (CFAR is scale-free, absolute dB values are not)")
        calib = 1.0
    inc = ov.get("incidence") or _pick(meta, "incidence_center", "incidence_angle_center")
    if inc is None:
        near, far = _pick(meta, "incidence_near"), _pick(meta, "incidence_far")
        inc = (near + far) / 2 if near is not None and far is not None else None
    res = [r for r in (_pick(meta, "range_resolution_center", "range_resolution"),
                       _pick(meta, "azimuth_resolution")) if r]
    alt = _pick(meta, "satellite_altitude", "average_altitude", "altitude")
    src = metric_grid(rasterio.open(tif))
    return SarScene(path, "iceye", src, t, scene_id=_pick(meta, "product_name", cast=str) or os.path.basename(tif),
                    scale=calib, amplitude=not str(meta.get("product_type", "")).upper().endswith("POWER"),
                    pol=_pick(meta, "polarization", cast=str) or "VV",
                    heading_deg=ov.get("heading") if ov.get("heading") is not None else _pick(meta, "heading", "satellite_heading"),
                    orbit_state=_pick(meta, "orbit_direction", cast=str),
                    look_side=_pick(meta, "look_side", cast=str) or "right", incidence_deg=inc,
                    altitude_m=alt if alt and alt > 1e5 else (alt * 1e3 if alt else None),
                    resolution_m=max(res) if res else None,
                    mode=_pick(meta, "acquisition_mode", "product_type", cast=str),
                    landmask=ov.get("landmask") or _find_landmask(tif), meta=meta)


def open_scene(path, **overrides):
    """Open a Blue Shadow SAR folder or an ICEYE product. Overrides (None = use metadata): time, heading, incidence,
    calib, landmask, pol."""
    path = path.rstrip("/")
    ov = {k: v for k, v in overrides.items() if v is not None}
    if os.path.isdir(path) and os.path.isfile(f"{path}/scene.json"):
        return open_blueshadow_dir(path, **ov)
    return open_iceye(path, **ov)
