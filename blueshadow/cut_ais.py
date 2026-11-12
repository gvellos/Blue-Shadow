"""Cut a daily DMA AIS file down to the minutes around a scene's acquisition time.

The Danish Maritime Authority publishes one CSV per day (aisdk-YYYY-MM-DD.csv, several GB). The pipelines only need
the messages near the image time, with a parsed `t` column.

  python -m blueshadow cut-ais data/ais/aisdk-2025-11-21.csv --scene data/s1/<pass>__<bbox>.S1
  python -m blueshadow cut-ais data/ais/aisdk-2025-11-21.csv --time 2025-11-21T17:00:50

The scene can be a SAR scene (Sentinel-1 folder or ICEYE product) or a Sentinel-2 .SAFE folder. The file is read in
chunks, so a full day never has to fit in memory. Writes data/ais/ais_<YYYYMMDD_HHMM>.csv unless --out is given.
"""
import argparse, os

import pandas as pd

TIME_COL, TIME_FMT = "# Timestamp", "%d/%m/%Y %H:%M:%S"


def scene_time(path):
    path = path.rstrip("/")
    if path.endswith(".SAFE"):
        from .pipeline_real import sensing_time
        return sensing_time(path)
    from . import sar
    return sar.open_scene(path).time


def cut(src, t, minutes, out, chunksize=1_000_000):
    lo, hi = t - pd.Timedelta(minutes=minutes), t + pd.Timedelta(minutes=minutes)
    parts, n = [], 0
    for chunk in pd.read_csv(src, chunksize=chunksize, low_memory=False):
        n += len(chunk)
        chunk["t"] = pd.to_datetime(chunk[TIME_COL], format=TIME_FMT, errors="coerce")
        parts.append(chunk[(chunk.t >= lo) & (chunk.t <= hi)])
        print(f"  read {n:,} messages...", end="\r", flush=True)
    a = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    a.to_csv(out, index=False)
    print(f"\n{len(a):,} of {n:,} messages within ±{minutes} min of {t} UTC "
          f"({a.MMSI.nunique() if len(a) else 0:,} vessels) -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ais", help="daily DMA AIS csv (aisdk-YYYY-MM-DD.csv)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="scene to take the acquisition time from")
    g.add_argument("--time", help="acquisition time UTC, e.g. 2025-11-21T17:00:50")
    ap.add_argument("--minutes", type=float, default=90, help="keep messages within this many minutes (default 90)")
    ap.add_argument("--out", help="output csv (default data/ais/ais_<YYYYMMDD_HHMM>.csv)")
    a = ap.parse_args()
    t = scene_time(a.scene) if a.scene else pd.Timestamp(a.time)
    t = t.tz_convert(None) if t.tzinfo else t
    out = a.out or f"data/ais/ais_{t:%Y%m%d_%H%M}.csv"
    cut(a.ais, t, a.minutes, out)
    print(f"Next: --ais {out}")


if __name__ == "__main__":
    main()
