"""Build small fake SAR scenes (+ DMA-style AIS, structures, ground truth) to smoke-test pipeline_sar.py offline.

  python -m blueshadow fake-sar --sensor s1      # fixtures/s1/   : night Sentinel-1 pass, 10 m UTM grid, descending
  python -m blueshadow fake-sar --sensor iceye   # fixtures/iceye/: ICEYE-style GRD, 2.5 m, amplitude DN, GCP georeferencing
  OUT=runs/fake_s1 python -m blueshadow sar --scene fixtures/s1/S1A_IW_GRDH_FAKE.S1 --ais fixtures/s1/ais.csv --structures fixtures/s1/structures.csv

Moving ships are painted where SAR would SHOW them (true position + Doppler azimuth shift from sar.py), while their
AIS reports the true position, so the fixture tests the shift correction: run with --no-az-shift to see fast crossing
ships fall out of the match gate. truth.csv lists the image positions (what the detector should find)."""
import argparse, json, math, os
import numpy as np, pandas as pd, rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as wt
from scipy import ndimage as ndi

from . import sar as S

SEA = 0.02          # sigma0 ~ -17 dB
ECHO_K = 0.3        # ship echo = ECHO_K * SEA * L^3 (m^3): 10x pipeline_sar's default; two real S1 night scenes measured 1-2.6


def paint(img, r, c, L, px, rng, blur_px=0.0):
    """Ship echo (hull L x L/6, random heading) with 4-look speckle, optionally blurred to the sensor resolution."""
    wd, os_ = max(L / 6, 2.0), 4
    n = int(math.ceil(L / px / 2)) + 3
    g = (np.arange((2 * n + 1) * os_) + 0.5) / os_ - (n + 0.5)
    cc, rr = np.meshgrid(g, g)
    ang = rng.uniform(0, math.pi)
    u = (cc * math.cos(ang) + rr * math.sin(ang)) * px
    v = (-cc * math.sin(ang) + rr * math.cos(ang)) * px
    cov = ((np.abs(u) <= L / 2) & (np.abs(v) <= wd / 2)).reshape(2 * n + 1, os_, 2 * n + 1, os_).mean(axis=(1, 3))
    if cov.sum() == 0:
        cov[n, n] = 1
    e = ECHO_K * SEA * L ** 3 * rng.lognormal(0, 0.25) / (px * px)
    patch = e * cov / cov.sum() * rng.gamma(4, 0.25, cov.shape)
    if blur_px:
        patch = ndi.gaussian_filter(patch, blur_px)
    img[r - n:r + n + 1, c - n:c + n + 1] += patch.astype(np.float32)


def ais_rows(ships, t_img, to_lonlat, rng):
    """DMA-style AIS rows. Each ship: dict(true_rc, L, mode, sog, cog). ok -> pings through image time; gap -> last ping
    40-66 min before; mismatch -> fresh AIS 800 m east of the ship; dark -> nothing."""
    rows = []

    def msg(mmsi, t, lon, lat, sog, cog, L, typ="Class A", name="TEST"):
        rows.append({"# Timestamp": t.strftime("%d/%m/%Y %H:%M:%S"), "Type of mobile": typ, "MMSI": mmsi, "Latitude": lat,
                     "Longitude": lon, "Navigational status": "Under way using engine", "ROT": 0, "SOG": sog, "COG": cog,
                     "Heading": cog, "IMO": "Unknown", "Callsign": "X", "Name": name, "Ship type": "Cargo", "Length": L,
                     "Width": L / 6})

    for k, s in enumerate(ships):
        if s["mode"] == "dark":
            continue
        lon, lat = to_lonlat(*s["true_rc"])
        if s["mode"] == "mismatch":
            lon += 800 / (111320 * math.cos(math.radians(lat)))
        ts = {"ok": np.arange(-1800, 601, 60 if s["L"] > 20 else 180), "gap": np.arange(-4000, -2400, 60),
              "mismatch": np.arange(-600, 601, 60)}[s["mode"]]
        v = s["sog"] * 0.5144
        for t in ts:
            dx, dy = v * t * math.sin(math.radians(s["cog"])), v * t * math.cos(math.radians(s["cog"]))
            msg(219000100 + k, t_img + pd.Timedelta(seconds=float(t)), lon + dx / (111320 * math.cos(math.radians(lat)))
                + rng.normal(0, 1e-5), lat + dy / 111320 + rng.normal(0, 1e-5), s["sog"], s["cog"], s["L"],
                "Class B" if s["L"] < 20 else "Class A", f"V{k}")
    return rows, msg


def finish(out, ships, apparent_lonlat, rows, msg, t_img, turbines_ll):
    lon, lat = turbines_ll[0]
    msg(992190001, t_img, lon, lat, 0, 0, 0, "AtoN", "WINDFARM N")
    msg(219999999, t_img, 20.0, 60.0, 5, 0, 50, "Class A", "FAR")
    df = pd.DataFrame(rows).sample(frac=1, random_state=1)
    df["t"] = pd.to_datetime(df["# Timestamp"], format="%d/%m/%Y %H:%M:%S")
    df.to_csv(f"{out}/ais.csv", index=False)
    pd.DataFrame([dict(lon=lo, lat=la, name=f"turbine {i}") for i, (lo, la) in enumerate(turbines_ll[1:], 1)]
                 ).to_csv(f"{out}/structures.csv", index=False)
    expect = {"ok": "matched", "gap": "unidentified", "mismatch": "unidentified", "dark": "unidentified"}
    pd.DataFrame([dict(lon=lo, lat=la, length_m=s["L"], mode=s["mode"], expect_status=expect[s["mode"]], expect_detect=True,
                       doppler_shift_m=round(s["shift_m"]))
                  for s, (lo, la) in zip(ships, apparent_lonlat)]).to_csv(f"{out}/truth.csv", index=False)


SHIPS = [  # (true row, true col, length m, mode, sog kn, cog deg) on a 1500 x 1500 grid; scaled for other grids
    (300, 600, 180, "ok", 12, 102), (500, 1100, 90, "ok", 14, 282), (800, 800, 45, "ok", 8, 10),
    (1100, 500, 60, "ok", 10, 120), (1250, 1200, 120, "ok", 0.2, 0), (700, 1300, 35, "ok", 16, 95),
    (950, 1150, 70, "gap", 6, 200), (1300, 700, 110, "dark", 9, 300), (400, 1000, 60, "mismatch", 7, 45),
    (600, 400, 28, "dark", 11, 80),
]


def build_s1(out="fixtures/s1"):
    rng = np.random.default_rng(5)
    N, PX = 1500, 10.0
    safe = f"{out}/S1A_IW_GRDH_FAKE.S1"
    os.makedirs(safe, exist_ok=True)
    crs, T = "EPSG:32632", from_origin(640000.0, 6170000.0, PX, PX)      # off Zealand, UTM 32N
    t_img = pd.Timestamp("2025-11-20 04:31:12")                           # 05:31 local: dark in November
    json.dump(dict(sensor="sentinel-1", id="S1A_IW_GRDH_FAKE", time_utc=t_img.isoformat(), pol="VV",
                   orbit_state="descending", heading_deg=192.0, look_side="right", incidence_deg=37.0,
                   resolution_m=20.0, mode="IW", product="fake RTC gamma0 (linear)"), open(f"{safe}/scene.json", "w"), indent=2)
    land_cols = (150 + 30 * np.sin(np.arange(N) / 120)).astype(int)
    land20 = np.zeros((N // 2, N // 2), np.uint8)
    for r in range(0, N, 2):
        land20[r // 2, :land_cols[r] // 2] = 1
    with rasterio.open(f"{safe}/landmask.tif", "w", driver="GTiff", width=N // 2, height=N // 2, count=1, dtype="uint8",
                       crs=crs, transform=from_origin(640000.0, 6170000.0, 20, 20)) as d:
        d.write(land20, 1)
    img = (SEA * (1 + 0.15 * ndi.gaussian_filter(rng.normal(size=(N, N)), 60) / 0.02).clip(0.6, 1.5)
           * rng.gamma(4, 0.25, (N, N))).astype(np.float32)
    for r in range(N):
        img[r, :land_cols[r]] = 0.15 * rng.gamma(2, 0.5, land_cols[r])
    img[:, -60:] = 0                                                       # swath edge: no data
    _write(f"{safe}/vv.tif", img, crs, T)
    scene = S.open_scene(safe)
    turbines = [(1250 + 60 * i, 300) for i in range(4)]
    for r, c in turbines:
        img[r, c] += 200 * SEA * 50
    to_ll = lambda r, c: tuple(x[0] for x in wt(crs, "EPSG:4326", [T.c + (c + 0.5) * PX], [T.f - (r + 0.5) * PX]))
    ships, apparent = [], []
    for r, c, L, mode, sog, cog in SHIPS:
        dx, dy = scene.azimuth_shift(sog, cog)
        ra, ca = int(round(r + dy / PX)), int(round(c + dx / PX))
        paint(img, ra, ca, L, PX, rng, blur_px=0.8)
        ships.append(dict(true_rc=(r, c), L=L, mode=mode, sog=sog, cog=cog, shift_m=math.hypot(dx, dy)))
        apparent.append(to_ll(ra, ca))
    _write(f"{safe}/vv.tif", img, crs, T)
    rows, msg = ais_rows(ships, t_img, to_ll, rng)
    finish(out, ships, apparent, rows, msg, t_img, [to_ll(r, c) for r, c in turbines])
    return safe, ships


def build_iceye(out="fixtures/iceye"):
    rng = np.random.default_rng(9)
    N, PX = 2000, 2.5                                                      # 5 km x 5 km stripmap-like crop
    os.makedirs(out, exist_ok=True)
    tif, name = f"{out}/ICEYE_X7_GRD_SM_FAKE.tif", "ICEYE_X7_GRD_SM_FAKE"
    lat0, lon0 = 55.62, 12.78                                              # Oresund
    dlat, dlon = PX / 111320, PX / (111320 * math.cos(math.radians(lat0)))
    to_ll = lambda r, c: (lon0 + (c + 0.5) * dlon, lat0 - (r + 0.5) * dlat)
    t0 = pd.Timestamp("2025-11-20 22:14:03")                               # tasked night acquisition
    calib = 2.0e-6
    json.dump(dict(product_name=name, product_type="GRD", acquisition_mode="stripmap", polarization="VV",
                   acquisition_start_utc=t0.isoformat() + "Z", acquisition_end_utc=(t0 + pd.Timedelta(seconds=4)).isoformat() + "Z",
                   calibration_factor=calib, orbit_direction="ASCENDING", look_side="right", incidence_center=28.0,
                   range_resolution=3.0, azimuth_resolution=3.0, satellite_altitude=560000.0),
              open(f"{out}/{name}.json", "w"), indent=2)
    sea = (SEA * rng.gamma(3, 1 / 3, (N, N))).astype(np.float32)
    land_c = 300
    sea[:, :land_c] = 0.15 * rng.gamma(2, 0.5, (N, land_c))
    ll_land = [to_ll(0, 0), to_ll(N, land_c)]
    with rasterio.open(f"{out}/{name}_landmask.tif", "w", driver="GTiff", width=40, height=40, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(ll_land[0][0] - dlon / 2, ll_land[0][1] + dlat / 2,
                                                             (N * dlon) / 40, (N * dlat) / 40)) as d:
        m = np.zeros((40, 40), np.uint8); m[:, :int(round(40 * land_c / N))] = 1
        d.write(m, 1)
    gcps = [GroundControlPoint(row=r, col=c, x=lon0 + c * dlon, y=lat0 - r * dlat)
            for r in np.linspace(0, N, 5) for c in np.linspace(0, N, 5)]

    def write(intensity):
        dn = np.sqrt(np.maximum(intensity, 0) / calib)
        with rasterio.open(tif, "w", driver="GTiff", width=N, height=N, count=1, dtype="uint16") as d:
            d.write(np.clip(dn, 1, 65535).astype(np.uint16), 1)
            d.gcps = (gcps, CRS.from_epsg(4326))

    write(sea)
    scene = S.open_scene(tif)
    ships, apparent = [], []
    scale = N / 1500 * 0.5                                                # spread fixture ships over the water part
    small = [(500, 1700, 6, "dark", 5, 30), (1600, 1500, 9, "ok", 7, 200), (1750, 900, 5, "dark", 4, 100)]
    layout = [(int(r * scale * 2 * 0.9 + 60), int(land_c + 150 + c * scale * 1.0), L, m, s, g) for r, c, L, m, s, g in SHIPS[:8]]
    turbines = [(150, 1500 + 80 * i) for i in range(3)]
    for r, c in turbines:
        sea[r - 1:r + 2, c - 1:c + 2] += 200 * SEA * 50 / 9
    for r, c, L, mode, sog, cog in layout + small:
        if mode == "gap":
            sog = 2.0           # keep its drift envelope inside this small scene from swallowing the dark ship
        dx, dy = scene.azimuth_shift(sog, cog)
        ra, ca = int(round(r + dy / PX)), int(round(c + dx / PX))
        if not (100 < ra < N - 100 and land_c + 100 < ca < N - 100):
            ra, ca = int(np.clip(ra, 100, N - 100)), int(np.clip(ca, land_c + 100, N - 100))
        paint(sea, ra, ca, L, PX, rng, blur_px=0.5)
        ships.append(dict(true_rc=(ra - dy / PX, ca - dx / PX), L=L, mode=mode, sog=sog, cog=cog, shift_m=math.hypot(dx, dy)))
        apparent.append(to_ll(ra, ca))
    write(sea)
    rows, msg = ais_rows(ships, t0 + pd.Timedelta(seconds=2), to_ll, rng)
    finish(out, ships, apparent, rows, msg, t0 + pd.Timedelta(seconds=2), [to_ll(r, c) for r, c in turbines])
    return tif, ships


def _write(path, img, crs, T):
    with rasterio.open(path, "w", driver="GTiff", width=img.shape[1], height=img.shape[0], count=1, dtype="float32",
                       crs=crs, transform=T, nodata=0) as d:
        d.write(img, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["s1", "iceye"], default="s1")
    a = ap.parse_args()
    path, ships = build_s1() if a.sensor == "s1" else build_iceye()
    shifts = ", ".join(f"{s['L']:.0f} m {s['mode']}: {s['shift_m']:.0f} m" for s in ships)
    print(f"fake {a.sensor} scene -> {path}\n  Doppler shifts painted in: {shifts}")
