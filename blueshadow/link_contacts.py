#!/usr/bin/env python3
"""Link Blue Shadow contacts across passes.

Two uses:

  # Link several real runs (each an OUT dir with all_detections.json + report.json), in any order:
  python -m blueshadow link out_day1 out_day2 out_day3 --out out_day3
      -> writes tracks.json into --out and injects `links` + `track_id` back into every run's
         all_detections.json and unidentified_contacts.json. Re-run build_ui.py on any run to see the links.

  # Self-contained demonstration with three fabricated passes (no data needed):
  python -m blueshadow link --demo --out out_demo && OUT=out_demo python -m blueshadow ui && open out_demo/ui.html
"""
import argparse, json, os
from datetime import datetime, timedelta, timezone
from . import linking as L


def load_run(d):
    recs = json.load(open(f"{d}/all_detections.json"))
    rep = json.load(open(f"{d}/report.json")) if os.path.exists(f"{d}/report.json") else {}
    scene = rep.get("scene") or os.path.basename(d.rstrip("/"))
    for r in recs:
        r["scene"] = scene
    return dict(dir=d, scene=scene, records=recs)


def earliest(run):
    return min((L.parse_time(r["time_utc"]) for r in run["records"]), default=datetime.max.replace(tzinfo=timezone.utc))


def link_runs(runs):
    runs = sorted(runs, key=earliest)
    history = []
    for run in runs:
        history = L.propose_links(run["records"], history)
    all_recs = [r for run in runs for r in run["records"]]
    return runs, L.build_tracks(all_recs)


def write_back(run, tracks):
    recs = run["records"]
    strip = lambda r: {k: v for k, v in r.items() if k != "scene"}
    json.dump([strip(r) for r in recs], open(f"{run['dir']}/all_detections.json", "w"), indent=2)
    queue = sorted([r for r in recs if r.get("status") == "unidentified"], key=lambda r: -r.get("priority", 0))
    json.dump([strip(r) for r in queue], open(f"{run['dir']}/unidentified_contacts.json", "w"), indent=2)
    rep_path = f"{run['dir']}/report.json"
    if os.path.exists(rep_path):
        rep = json.load(open(rep_path))
        rep["tracks"] = tracks
        rep["links_proposed"] = sum(len(r.get("links", [])) for r in recs)
        json.dump(rep, open(rep_path, "w"), indent=2)


# ---------------------------------------------------------------- demonstration
def make_demo(out):
    """Three daily passes over the same patch of sea. Vessel A drifts between passes (recurring dark contact);
    vessel B is a dark candidate on day 1/2 then appears on AIS on day 3 (tentative resolution); C is unrelated."""
    os.makedirs(f"{out}/crops", exist_ok=True)
    t0 = datetime(2026, 11, 5, 10, 30, tzinfo=timezone.utc)
    runs = []
    tracks_truth = dict(
        A=[(0, 55.40, 14.00, 95, "DARK_CANDIDATE", None), (1, 55.47, 14.12, 98, "DARK_CANDIDATE", None),
           (2, 55.55, 14.26, 96, "DARK_CANDIDATE", None)],                                   # steady NE transit, dark
        B=[(0, 55.30, 14.30, 60, "POSSIBLE_AIS_GAP", None), (1, 55.33, 14.33, 62, "DARK_CANDIDATE", None),
           (2, 55.36, 14.35, 61, "AIS_MATCH", 211334455)],                                   # dark, then turns on AIS
        C=[(1, 55.60, 13.80, 22, "SMALL_BELOW_LIMIT", None)])                                # one-off, no link
    truth_map = {}                                                                            # contact id -> true vessel
    for day in range(3):
        t = t0 + timedelta(days=day, minutes=7 * day)
        recs, n = [], 0
        for name, obs in tracks_truth.items():
            for (d, lat, lon, Lm, code, mmsi) in obs:
                if d != day:
                    continue
                n += 1
                matched = code == "AIS_MATCH"
                rec = dict(id=f"BS-{t:%Y%m%d}-{n:04d}", time_utc=t.isoformat(), lat=lat, lon=lon,
                           confidence=0.8, confidence_breakdown=dict(contrast=0.8, size_plausibility=1.0, clutter_distance=0.9),
                           est_length_m=Lm, est_length_uncertainty_m=10 + round(0.3 * Lm), snr_db=12.0,
                           status="matched" if matched else "unidentified", reason_code=code,
                           map_xy=None, links=[])
                truth_map[rec["id"]] = name
                if matched:
                    rec["evidence"] = dict(mmsi=mmsi, name="REVEALED", dist_m=40, ais_method="interpolated", ais_age_s=30)
                else:
                    rec.update(reason_text="Demo contact.", evidence={}, in_sensitive_zone=False,
                               priority=round(L.P_CAP * 0.8, 3), analyst_state="new", nearby_ais=[])
                recs.append(rec)
        d = f"{out}/_pass{day}"
        os.makedirs(d, exist_ok=True)
        json.dump(recs, open(f"{d}/all_detections.json", "w"), indent=2)
        json.dump(dict(real=False, scene=f"DEMO pass {day + 1} ({t:%Y-%m-%d})", sensing_time_utc=t.isoformat(),
                       detections=len(recs), matched=sum(r["status"] == "matched" for r in recs),
                       unidentified=sum(r["status"] == "unidentified" for r in recs),
                       reason_breakdown={}, size_bands=[], detection_limit_curve=[], map_size=[1500, 1500]),
                  open(f"{d}/report.json", "w"), indent=2)
        runs.append(load_run(d))

    runs, tracks = link_runs(runs)
    for run in runs:
        write_back(run, tracks)
    # merge all passes into the --out dir so build_ui.py renders one cross-pass map
    all_recs = [r for run in runs for r in run["records"]]
    strip = lambda r: {k: v for k, v in r.items() if k != "scene"}
    queue = sorted([r for r in all_recs if r.get("status") == "unidentified"], key=lambda r: -r.get("priority", 0))
    json.dump([strip(r) for r in all_recs], open(f"{out}/all_detections.json", "w"), indent=2)
    json.dump([strip(r) for r in queue], open(f"{out}/unidentified_contacts.json", "w"), indent=2)
    json.dump(dict(real=False, scene="Cross-pass demo (3 passes)", detections=len(all_recs),
                   matched=sum(r["status"] == "matched" for r in all_recs), unidentified=len(queue),
                   reason_breakdown={c: sum(r.get("reason_code") == c for r in queue) for c in {r.get("reason_code") for r in queue}},
                   size_bands=[], detection_limit_curve=[], tracks=tracks, links_proposed=sum(len(r.get("links", [])) for r in all_recs),
                   map_size=[1500, 1500],
                   note="Cross-pass linking demo: three daily passes. Links are tentative (p capped) and never assign identity."),
              open(f"{out}/report.json", "w"), indent=2)
    json.dump(truth_map, open(f"{out}/link_truth.json", "w"), indent=2)
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="run output directories, each with all_detections.json")
    ap.add_argument("--out", default="runs/links", help="where to write tracks.json (and demo outputs)")
    ap.add_argument("--demo", action="store_true", help="fabricate three passes and link them")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo:
        tracks = make_demo(a.out)
    else:
        if len(a.dirs) < 2:
            ap.error("give at least two run directories, or use --demo")
        runs, tracks = link_runs([load_run(d) for d in a.dirs])
        for run in runs:
            write_back(run, tracks)
    json.dump(tracks, open(f"{a.out}/tracks.json", "w"), indent=2)
    print(f"{len(tracks)} cross-pass track(s) -> {a.out}/tracks.json")
    for t in tracks:
        tag = f"resolved MMSI {t['resolved_mmsi']}" if t["resolved_mmsi"] else "unresolved"
        print(f"  {t['track_id']}: {t['n_observations']} obs, {t['first_seen'][:16]} -> {t['last_seen'][:16]}, "
              f"max p={t['max_link_p']}, {tag}")


if __name__ == "__main__":
    main()
