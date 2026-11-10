"""Build a tiny fake Sentinel-2 L2A SAFE (GeoTIFF instead of JP2) + DMA-style AIS csv to smoke-test pipeline_real.py.
Usage: python make_fake_scene.py && OUT=./out_fake python pipeline_real.py --safe fake/*.SAFE --ais fake/ais.csv --structures fake/structures.csv --kseed 8
The fake AIS already carries a parsed "t" column, as ais_window.csv written by inspect_data.py does."""
import math, os
import numpy as np, pandas as pd, rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as wt

rng = np.random.default_rng(3)
N10, PX = 2400, 10.0
safe = "fake/S2A_MSIL2A_20250927T103041_N0511_R108_T33UUB_20250927T141918.SAFE"
gran = f"{safe}/GRANULE/L2A_T33UUB_A000001_20250927T103041"
for d in ("R10m", "R20m"):
    os.makedirs(f"{gran}/IMG_DATA/{d}", exist_ok=True)
open(f"{gran}/MTD_TL.xml", "w").write("<x><SENSING_TIME metadataLevel=\"Standard\">2025-09-27T10:30:55.123Z</SENSING_TIME></x>")

crs = "EPSG:32633"
x0, y0 = 360000.0, 6180000.0          # UTM top-left
T10 = from_origin(x0, y0, PX, PX)
T20 = from_origin(x0, y0, 20, 20)

img = (1100 + rng.normal(0, 25, (N10, N10))).astype(np.float32)
scl = np.full((N10 // 2, N10 // 2), 6, np.uint8)
# land strip on the left (with ragged coast), a bright pier connected to it
land_cols = (300 + 40 * np.sin(np.arange(N10) / 90)).astype(int)
for r in range(N10):
    img[r, :land_cols[r]] = 2500 + rng.normal(0, 300, land_cols[r])
    scl[r // 2, :land_cols[r] // 2] = 4
img[1200:1206, 300:420] = 3500; scl[600:603, 150:210] = 7
# cloud patch
img[200:420, 1800:2050] += 4000; scl[100:210, 900:1025] = 9
# thin cirrus region
scl[1000:1200, 600:1200] = 10; img[2000:2400, 1200:2400] += 150

def put_ship(r, c, L, excess=1800):
    wd = max(L / 5, 2)
    ang = rng.uniform(0, math.pi)
    n = int(L / PX) + 3
    rr, cc = np.mgrid[-n:n + 1, -n:n + 1]
    u = (cc * math.cos(ang) + rr * math.sin(ang)) * PX
    v = (-cc * math.sin(ang) + rr * math.cos(ang)) * PX
    m = (np.abs(u) <= L / 2) & (np.abs(v) <= wd / 2)
    if L < 12: m[n, n] = True
    sub = img[r - n:r + n + 1, c - n:c + n + 1]
    sub[m] += excess * (0.3 if L < 10 else 1.0)

def rc_to_lonlat(r, c):
    e, n = T10 * (c + 0.5, r + 0.5)
    lon, lat = wt(crs, "EPSG:4326", [e], [n])
    return lon[0], lat[0]

ships = [  # (row, col, L, mode)
    (600, 900, 180, "ok"), (800, 1500, 90, "ok"), (1500, 1200, 45, "ok"), (1800, 700, 25, "ok"),
    (400, 1300, 12, "ok"), (1000, 2200, 8, "ok"), (1300, 1900, 60, "gap"), (2100, 1000, 120, "dark"),
    (1700, 2100, 30, "ok"), (900, 600, 15, "ok"), (2200, 1800, 70, "mismatch"),
]
for r, c, L, _ in ships:
    put_ship(r, c, L)
# turbines (static, one with AtoN AIS, others only in structures.csv) + a small boat under the cloud
turbines = [(1400 + 60 * i, 1600) for i in range(4)]
for r, c in turbines:
    img[r, c] += 5000
put_ship(300, 1900, 40)   # under cloud -> should be masked

with rasterio.open(f"{gran}/IMG_DATA/R10m/T33UUB_20250927T103041_B08_10m.tif", "w", driver="GTiff", width=N10, height=N10,
                   count=1, dtype="uint16", crs=crs, transform=T10) as dst:
    dst.write(np.clip(img, 0, 65535).astype(np.uint16), 1)
with rasterio.open(f"{gran}/IMG_DATA/R20m/T33UUB_20250927T103041_SCL_20m.tif", "w", driver="GTiff", width=N10 // 2, height=N10 // 2,
                   count=1, dtype="uint8", crs=crs, transform=T20) as dst:
    dst.write(scl, 1)
vis = np.clip(img * 0.8, 0, 65535).astype(np.uint16)
for b in ("B04", "B03", "B02"):
    with rasterio.open(f"{gran}/IMG_DATA/R10m/T33UUB_20250927T103041_{b}_10m.tif", "w", driver="GTiff", width=N10, height=N10,
                       count=1, dtype="uint16", crs=crs, transform=T10) as dst:
        dst.write(vis, 1)

# ---- AIS
t_img = pd.Timestamp("2025-09-27 10:30:55")
rows = []
def msg(mmsi, t, lon, lat, sog, cog, L, typ="Class A", name="TEST"):
    rows.append({"# Timestamp": t.strftime("%d/%m/%Y %H:%M:%S"), "Type of mobile": typ, "MMSI": mmsi, "Latitude": lat,
                 "Longitude": lon, "Navigational status": "Under way using engine", "ROT": 0, "SOG": sog, "COG": cog,
                 "Heading": cog, "IMO": "Unknown", "Callsign": "X", "Name": name, "Ship type": "Cargo", "Length": L, "Width": L / 6})

for k, (r, c, L, mode) in enumerate(ships):
    mmsi = 219000100 + k
    if mode == "dark":
        continue
    lon, lat = rc_to_lonlat(r, c)
    sog, cog = rng.uniform(3, 12), rng.uniform(0, 360)
    v = sog * 0.5144
    if mode == "ok":
        ts = np.arange(-1800, 601, 60 if L > 20 else 180)
    elif mode == "gap":
        ts = np.arange(-4000, -2400, 60)
    else:  # mismatch: fresh AIS 800 m away
        ts = np.arange(-600, 601, 60)
        lon, lat = rc_to_lonlat(r, c + 80)
    for t in ts:
        dx, dy = v * t * math.sin(math.radians(cog)), v * t * math.cos(math.radians(cog))
        la = lat + dy / 111320 + rng.normal(0, 1e-5)
        lo = lon + dx / (111320 * math.cos(math.radians(lat))) + rng.normal(0, 1e-5)
        msg(mmsi, t_img + pd.Timedelta(seconds=float(t)), lo, la, sog, cog, L, "Class B" if L < 20 else "Class A", f"V{k}")
# a junk jump message for vessel 0
lon, lat = rc_to_lonlat(600, 900)
msg(219000100, t_img + pd.Timedelta(seconds=-5), lon + 0.5, lat, 5, 90, 180)
# AtoN on first turbine, base station, far-away vessel
lon, lat = rc_to_lonlat(*turbines[0]); msg(992190001, t_img, lon, lat, 0, 0, 0, "AtoN", "WINDFARM N")
msg(2190047, t_img, lon, lat + 0.3, 0, 0, 0, "Base Station", "BS")
msg(219999999, t_img, 20.0, 60.0, 5, 0, 50, "Class A", "FAR")
df = pd.DataFrame(rows).sample(frac=1, random_state=1)
df["t"] = pd.to_datetime(df["# Timestamp"], format="%d/%m/%Y %H:%M:%S")
df.to_csv("fake/ais.csv", index=False)
pd.DataFrame([dict(lon=rc_to_lonlat(r, c)[0], lat=rc_to_lonlat(r, c)[1], name=f"turbine {i}") for i, (r, c) in enumerate(turbines[1:])]
             ).to_csv("fake/structures.csv", index=False)

# ---- ground truth for eval.py: every injected vessel, its length, and the status the pipeline should assign.
# ok -> on AIS at image time (matched); gap/mismatch/dark -> no usable AIS at image time (unidentified).
EXPECT = {"ok": "matched", "gap": "unidentified", "mismatch": "unidentified", "dark": "unidentified"}
truth = []
for r, c, L, mode in ships:
    lon, lat = rc_to_lonlat(r, c)
    truth.append(dict(lon=lon, lat=lat, length_m=L, mode=mode, expect_status=EXPECT[mode], expect_detect=True))
# the ship painted under the cloud patch must be masked away, i.e. NOT detected
lon, lat = rc_to_lonlat(300, 1900)
truth.append(dict(lon=lon, lat=lat, length_m=40, mode="masked", expect_status="", expect_detect=False))
pd.DataFrame(truth).to_csv("fake/truth.csv", index=False)
print("fake dataset written; dark ship at", rc_to_lonlat(2100, 1000), "| truth: fake/truth.csv")
