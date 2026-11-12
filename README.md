# Blue Shadow: explained dark-vessel contacts

Detect vessels in Sentinel imagery, fuse with Danish Maritime Authority AIS interpolated to image time, and hand an
analyst a queue of **unidentified contacts**. Each has a unique ID, location, timestamp, confidence score, image crop,
estimated size, and a **reason code** explaining why no AIS match was found. Performance is reported by vessel size band,
with the ~20 m Sentinel resolution limit made visible.

## Pipeline
1. **Detect**: CFAR (two-parameter, hysteresis) with land, cloud and coast masks.
2. **AIS**: clean, interpolate or dead-reckon to image time, with an uncertainty radius that grows with the gap.
3. **Match**: gated Hungarian assignment.
4. **Explain** each unmatched contact: `STATIC_STRUCTURE`, `NO_AIS_COVERAGE`, `NEAR_COAST_CLUTTER`, `POSSIBLE_AIS_GAP`,
   `AIS_MISMATCH`, `SMALL_BELOW_LIMIT`, `DARK_CANDIDATE` (heuristics with evidence, not proofs).
5. **Report** recall per size band with 95% intervals; AIS is a biased reference (cooperative vessels only).
6. **Analyst UI**: single-file HTML map, queue and contact cards (Investigate / Keep unresolved / Dismiss).

## Quick start
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Demo on a simulated scene
OUT=./out python pipeline.py && OUT=./out python build_ui.py && open out/ui.html
```

## Real data (Sentinel-2 L2A + DMA AIS)
1. Download an S2 L2A product from the Copernicus Data Space (https://dataspace.copernicus.eu) and the DMA AIS CSV for the
   same day (https://www.dma.dk/safety-at-sea/navigational-information/ais-data).
   *Or* skip the manual download and pull just the area + bands you need from Microsoft Planetary Computer (no account):
   `python fetch_scene.py --bbox 12.40 55.40 13.10 55.90 --date 2025-09-27` picks the least-cloudy, best-covered scene
   within `--days` of the date and writes a small `s2_extracted/<id>.SAFE` (it reads only the bbox window of
   B08/B04/B03/B02 + SCL via HTTP range requests, so no 1 GB download). Scene selection rejects near-empty swath-edge
   tiles (`--min-coverage`, default 40% real data) so you don't get a mostly-nodata tile. Keep the bbox over water and
   inside one S2 tile. Requires `pip install pystac-client planetary-computer` (add `shapely` for bbox-overlap ranking).
2. `python inspect_data.py <S2 L2A .SAFE.zip> <aisdk-YYYY-MM-DD.csv>` extracts the scene, reports cloud/water and AIS counts,
   and writes `ais_window.csv`.
3. `OUT=./out python pipeline_real.py --osm` (raise `--kseed` if there are too many false alarms), then
   `OUT=./out python build_ui.py`. Useful flags: `--structures my.csv` (lon,lat[,name] of turbines/buoys),
   `--mask-cirrus`, `--inject-pct 15`, `--no-inject`, `--sensitive lon0,lat0,lon1,lat1`.

What the real-data path does:
- Masks from the 20 m scene classification: land/cloud blobs smaller than a ship-sized area are *not* masked (Sen2Cor often
  classes bright ships as cloud or unclassified); thin cirrus is kept but flagged and discounts confidence.
- Mask-aware, block-wise CFAR on B08 (the full tile never sits in memory).
- AIS cleaning (9-digit MMSI, duplicates, >60 kn jump outliers, SOG/COG sentinels), projection through the tile's UTM grid,
  interpolation to the granule sensing time from `MTD_TL.xml`.
- Static-structure catalogue from AIS AtoN + optional csv + OpenStreetMap (`--osm`: turbines, platforms, buoys, lights).
- Detection-limit curve from synthetic boats pasted into *real* clear-water patches of the scene (sub-pixel coverage model,
  hull contrast taken from matched 20-100 m vessels). Reported as measured-but-simulated.
- Extra outputs: `contacts.geojson` (detections + interpolated AIS), NIR and RGB crops, nearest AIS tracks per contact.

Smoke test without real data: `python make_fake_scene.py && OUT=./out_fake python pipeline_real.py --safe fake/*.SAFE --ais fake/ais.csv --structures fake/structures.csv --kseed 8`.

Data files are git-ignored. Check the DMA and Copernicus terms of use before redistributing anything.

## Night and cloud: SAR (Sentinel-1 + ICEYE)
Sentinel-2 is optical, so it sees nothing at night (and it only passes at ~10:30 local time anyway). Radar sends its own
pulse, so it sees in darkness and through cloud. `pipeline_sar.py` runs the same detect → match → explain → report chain
on SAR and writes the same outputs, so `build_ui.py`, `link_contacts.py` and `eval.py` work unchanged. Two sensors are
supported, and they're meant to be used together ("tip and cue"):

- **Sentinel-1**: free, systematic. It passes Denmark around 05:30 and 17:30 local time, both dark from about October to
  March. ~20 m resolution, so it finds ships over roughly 20 m. Use it to search the whole area.
- **ICEYE**: commercial, tasked (you order a place and time). 0.5–3 m resolution, so it also sees small boats. Use it to
  confirm a Sentinel-1 `DARK_CANDIDATE` at its predicted position. Cross-pass linking then ties the two sightings together.

```
# Sentinel-1: pick a night pass and pull the bbox window (VV/VH + ESA WorldCover land mask) from Planetary Computer
python fetch_scene.py --sensor s1 --bbox 12.55 55.55 12.85 55.75 --date 2025-11-20 --days 6 --night
OUT=./out_s1 python pipeline_sar.py --scene s1_extracted/<id>.S1 --ais ais_window.csv --osm
OUT=./out_s1 python build_ui.py

# ICEYE: point it at the GRD GeoTIFF (its .json/.xml metadata sidecar next to it); build a land mask once
python fetch_scene.py --landmask-for path/to/ICEYE_..._GRD.tif
OUT=./out_iceye python pipeline_sar.py --scene path/to/ICEYE_..._GRD.tif --ais ais_window.csv --osm
```

What changes for radar (`sar.py` loads the scene, `pipeline_sar.py` runs it):
- **One scene interface.** `sar.open_scene()` returns linear backscatter on a north-up UTM grid with square pixels,
  warping on the fly when a product is in lon/lat or radar geometry (GCPs, fitted affine; it warns if the fit is poor).
  ICEYE intensity = `calibration_factor × DN²`. Missing or wrong metadata can be overridden with `--time`, `--heading`,
  `--incidence`, `--calib`, `--landmask`.
- **Doppler shift.** A ship moving towards or away from the radar is imaged in the wrong place along the flight direction:
  about 110 s × its slant-range speed for Sentinel-1, which is often 300–500 m. Each AIS position is moved by the
  predicted shift before matching. On the fake Sentinel-1 scene this is what turns 2 matches into 6;
  `--no-az-shift` lets you compare.
- **CFAR in metres.** The guard and background windows are fixed at 410 m and 910 m, so a 2.5 m ICEYE grid gets
  windows ~4× wider in pixels than a 10 m Sentinel-1 grid. The detector is **two-pass**: bright targets are left out of
  the second background estimate, so a ship next to another ship (a convoy, an anchorage, a wind farm) isn't hidden.
- **Automatic threshold.** The seed threshold comes from the speckle measured on open water (gamma ENL fitted to the
  99.9th percentile) and a false-alarm target (`--fa-rate`, per 100 km²: 1 for Sentinel-1, 0.1 for ICEYE). Use `--kseed`
  to set it by hand.
- **Land mask.** Comes from ESA WorldCover; there's no cloud mask, because radar sees through cloud. The small-boat limit
  is about one resolution cell: 20 m for Sentinel-1, ~4 m for ICEYE. The size bands get finer when the sensor can
  resolve small boats.
- **Detection limit.** The curve comes from synthetic ship echoes pasted into real clear-water patches of the scene. The
  echo strength is calibrated from matched AIS vessels.

Test without real data (the fixtures paint moving ships at their Doppler-shifted positions):
```
python eval.py --build --sar s1      # fake night Sentinel-1 pass
python eval.py --build --sar iceye   # fake ICEYE GRD (2.5 m, amplitude DN, GCP georeferencing)
```

Limitations of the SAR path: azimuth ambiguities ("ghosts") of very bright ships and wind-farm side-lobes can still
trigger false alarms. The Doppler shift uses a flat-earth geometry with the nominal heading (about 10 % accuracy). The
ICEYE loader follows ICEYE's documented metadata keys but has only been tested on the fixture, not on a real delivery,
so check the printed scene summary. ICEYE data isn't free; ESA's third-party missions programme gives researchers
access on proposal.

## Cross-pass contact linking
A dark contact in one scene may be the same vessel seen in an earlier pass. `linking.py` proposes **tentative** links
(never an identity) from checkable physics: implied speed (distance / time gap), length agreement, and a time-gap decay
tuned to the Sentinel revisit. Two link kinds: `recurring_contact` (dark → dark) and `resolved_by_ais` (an earlier dark
candidate that appears on AIS later — a possible identity to verify). Strongest links win a one-to-one assignment, chaining
contacts into tracks (`TRK-*`); probabilities are capped, so links stay advisory.

```
# Operational: accumulate a history file across runs; this pass links back to prior ones
OUT=./out1 python pipeline_real.py --history history.json ...   # pass 1
OUT=./out2 python pipeline_real.py --history history.json ...   # pass 2 links to pass 1

# Retroactively link several existing runs and build tracks.json:
python link_contacts.py out1 out2 out3 --out out3

# Self-contained demo (no data): three passes, a recurring dark contact and one resolved by AIS
python link_contacts.py --demo --out out_demo && OUT=out_demo python build_ui.py && open out_demo/ui.html
```
The UI shows candidate links on each contact card and a **Tracks** tab with the cross-pass chains.

## Evaluation
`eval.py` turns "looks right" into numbers so tuning can be measured and regressions caught. It scores against
known ground truth that the fixtures emit (`make_fake_scene.py` → `fake/truth.csv`; `link_contacts.py --demo`
→ `link_truth.json`):

- **Detection:** recall, precision, false-alarm count, length MAE, and status-classification accuracy per mode
  (dark / gap / mismatch / ok), including that a vessel painted under a cloud is correctly masked away.
- **Linking:** pairwise precision / recall / F1 of the proposed tracks vs. the true vessel groupings.

```
python eval.py --build            # build the fixture, run the pipeline, score detection + linking
python eval.py --run-dir out_fake # score an existing run; --detect-only / --link-only; --json metrics.json
python eval.py --build --ui out_eval  # also write a browsable UI with an Evaluation tab -> out_eval/ui.html
```
It exits non-zero if any metric falls below the thresholds in `THRESHOLDS`, so it doubles as a CI / pre-commit gate.
The `--ui` flag adds an **Evaluation** tab to the normal UI, showing the detection and linking metrics (with pass/fail
colouring and the per-scenario breakdown) alongside the map, queue and tracks.

## Limitations
- Real-data thresholds are untuned; expect sun glint and whitecaps to drive false alarms (`--kseed`). Tested on a fabricated mini-scene.
- SAR path (Sentinel-1 / ICEYE) handles the Doppler azimuth shift but not azimuth ambiguities; see the SAR section.
- S2 band-offset speed estimate and calibrated confidence are not implemented.
