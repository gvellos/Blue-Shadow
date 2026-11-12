#!/usr/bin/env python3
"""Evaluation harness for Blue Shadow: turn "looks right" into numbers.

Two things are scored against known ground truth, so tuning (CFAR, AIS matching, link scoring) can be
measured and regressions caught:

  detection  -- run pipeline_real.py on the synthetic scene (make_fake_scene.py writes fixtures/s2/truth.csv with
                every injected vessel, its length and the status it *should* get) and compute recall,
                precision, false-alarm count, length error (MAE) and status-classification accuracy.
  linking    -- run the cross-pass demo (link_contacts.py --demo writes link_truth.json mapping each
                contact to its true vessel) and compute how well the proposed tracks recover the true
                groupings (pairwise precision / recall / F1).

Usage:
  python -m blueshadow eval --build              # build the fixture, run the pipeline, then score everything
  python -m blueshadow eval --run-dir out_fake   # score an existing pipeline run against fixtures/s2/truth.csv
  python -m blueshadow eval --detect-only | --link-only
  python -m blueshadow eval --build --json metrics.json   # also dump machine-readable metrics
  python -m blueshadow eval --build --sar s1 | --sar iceye   # score the radar path (pipeline_sar.py) on a fake SAR scene

Exit code is non-zero if any metric falls below the thresholds in THRESHOLDS (a CI / pre-commit gate).
"""
import argparse, json, os, subprocess, sys, tempfile
from itertools import combinations

from . import linking as L

PY = sys.executable
MATCH_RADIUS_M = 120.0        # a detection within this distance of a truth point is the same vessel
THRESHOLDS = dict(            # minimum acceptable metrics; below any of these, eval.py exits 1
    detect_recall=0.80, detect_precision=0.80, status_accuracy=0.70,
    link_f1=0.80,
)


# ---------------------------------------------------------------- detection
def _load_truth(path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(lon=float(r["lon"]), lat=float(r["lat"]), length_m=float(r["length_m"]),
                             mode=r["mode"], expect_status=r["expect_status"],
                             expect_detect=str(r["expect_detect"]).lower() in ("true", "1")))
    return rows


def score_detection(run_dir, truth_csv):
    truth = _load_truth(truth_csv)
    dets = json.load(open(f"{run_dir}/all_detections.json"))
    # turbines are scored elsewhere; SAR side-lobe artifacts are flagged, never queued, so they aren't vessel claims
    vessels = [d for d in dets if d.get("status") not in ("static_structure", "artifact")]

    # greedy nearest-neighbour matching of each truth point to one detection
    used, matches = set(), {}
    for i, t in enumerate(truth):
        best, best_d = None, MATCH_RADIUS_M
        for j, d in enumerate(vessels):
            if j in used or d.get("lat") is None:
                continue
            km = L.haversine_km(t["lat"], t["lon"], d["lat"], d["lon"])
            if km * 1000 <= best_d:
                best, best_d = j, km * 1000
        if best is not None:
            used.add(best)
            matches[i] = best

    want = [i for i, t in enumerate(truth) if t["expect_detect"]]
    tp = sum(1 for i in want if i in matches)
    fn = len(want) - tp
    # false alarms: vessel detections matching no truth point, and detections of a should-be-masked vessel
    masked_hits = sum(1 for i, t in enumerate(truth) if not t["expect_detect"] and i in matches)
    fp = (len(vessels) - len(used)) + masked_hits

    recall = tp / len(want) if want else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0

    # length error + status accuracy over matched expected-present vessels
    len_err, status_ok, status_n = [], 0, 0
    per_mode = {}
    for i in want:
        t = truth[i]
        pm = per_mode.setdefault(t["mode"], dict(n=0, found=0, status_ok=0))
        pm["n"] += 1
        if i in matches:
            pm["found"] += 1
            d = vessels[matches[i]]
            if d.get("est_length_m"):
                len_err.append(abs(d["est_length_m"] - t["length_m"]))
            status_n += 1
            if d.get("status") == t["expect_status"]:
                status_ok += 1
                pm["status_ok"] += 1
    mae = sum(len_err) / len(len_err) if len_err else 0.0
    status_acc = status_ok / status_n if status_n else 1.0

    return dict(truth_vessels=len(want), detections=len(vessels), tp=tp, fp=fp, fn=fn,
                masked_leaks=masked_hits, recall=round(recall, 3), precision=round(precision, 3),
                length_mae_m=round(mae, 1), status_accuracy=round(status_acc, 3),
                per_mode={m: dict(detected=f"{v['found']}/{v['n']}", status_ok=f"{v['status_ok']}/{v['n']}")
                          for m, v in sorted(per_mode.items())})


# ---------------------------------------------------------------- linking
def score_linking(out_dir):
    truth = json.load(open(f"{out_dir}/link_truth.json"))          # contact id -> true vessel
    tracks = json.load(open(f"{out_dir}/tracks.json"))
    # predicted grouping: contact id -> track id (contacts in no track are singletons)
    pred = {}
    for t in tracks:
        for obs in t.get("observations", t.get("members", [])):
            cid = obs.get("id") if isinstance(obs, dict) else obs
            pred[cid] = t["track_id"]

    ids = [c for c in truth if truth[c] is not None]
    same_true = lambda a, b: truth[a] == truth[b]
    same_pred = lambda a, b: pred.get(a, f"_{a}") == pred.get(b, f"_{b}")   # unlinked -> its own singleton

    tp = fp = fn = tn = 0
    for a, b in combinations(ids, 2):
        st, sp = same_true(a, b), same_pred(a, b)
        tp += st and sp
        fp += (not st) and sp
        fn += st and (not sp)
        tn += (not st) and (not sp)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(contacts=len(ids), tracks=len(tracks), pairs_tp=tp, pairs_fp=fp, pairs_fn=fn,
                pair_precision=round(precision, 3), pair_recall=round(recall, 3), link_f1=round(f1, 3))


# ---------------------------------------------------------------- drivers
SAR_FIXTURES = {"s1": "fixtures/s1/S1A_IW_GRDH_FAKE.S1", "iceye": "fixtures/iceye/ICEYE_X7_GRD_SM_FAKE.tif"}


def build_fixture_run(out, sar=None):
    env = dict(os.environ, OUT=out)
    if sar:
        print(f"Building synthetic {sar} SAR scene (make_fake_sar.py)...", flush=True)
        subprocess.run([PY, "-m", "blueshadow.make_fake_sar", "--sensor", sar], check=True)
        print("Running pipeline_sar.py on the scene...", flush=True)
        subprocess.run([PY, "-m", "blueshadow.pipeline_sar", "--scene", SAR_FIXTURES[sar], "--ais", f"fixtures/{sar}/ais.csv",
                        "--structures", f"fixtures/{sar}/structures.csv"], check=True, env=env)
        return out
    print("Building synthetic scene (make_fake_scene.py)...", flush=True)
    subprocess.run([PY, "-m", "blueshadow.make_fake_scene"], check=True)
    safe = subprocess.run("ls -d fixtures/s2/*.SAFE", shell=True, capture_output=True, text=True).stdout.strip()
    print("Running pipeline_real.py on the scene...", flush=True)
    subprocess.run([PY, "-m", "blueshadow.pipeline_real", "--safe", safe, "--ais", "fixtures/s2/ais.csv",
                    "--structures", "fixtures/s2/structures.csv", "--kseed", "8"], check=True, env=env)
    return out


def build_link_demo(workdir):
    out = os.path.join(workdir, "out_linkdemo")
    print("Running cross-pass linking demo (link_contacts.py --demo)...", flush=True)
    subprocess.run([PY, "-m", "blueshadow.link_contacts", "--demo", "--out", out], check=True)
    return out


def render_ui(run_dir, metrics, failures):
    """Inject the scored metrics into the run's report.json and build the browsable UI for it."""
    rep_path = f"{run_dir}/report.json"
    rep = json.load(open(rep_path))
    rep["evaluation"] = dict(detection=metrics.get("detection"), linking=metrics.get("linking"),
                             thresholds=THRESHOLDS, passed=not failures, failures=failures)
    json.dump(rep, open(rep_path, "w"), indent=2)
    print("Building UI (build_ui.py)...", flush=True)
    subprocess.run([PY, "-m", "blueshadow.build_ui"], check=True, env=dict(os.environ, OUT=run_dir))
    return f"{run_dir}/ui.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build the fixture + run the pipeline before scoring")
    ap.add_argument("--run-dir", help="score detection against an existing pipeline run dir (uses fixtures/s2/truth.csv)")
    ap.add_argument("--sar", choices=sorted(SAR_FIXTURES), help="score the SAR path on a fake s1 or iceye scene instead")
    ap.add_argument("--truth", default=None, help="ground truth csv (default fixtures/s2/truth.csv, or fixtures/<sar>/truth.csv)")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--link-only", action="store_true")
    ap.add_argument("--json", help="also write all metrics to this file")
    ap.add_argument("--ui", help="write a browsable UI (adds an Evaluation tab) to this dir, e.g. --ui out_eval")
    a = ap.parse_args()

    do_detect = not a.link_only
    do_link = not a.detect_only
    metrics, failures = {}, []
    workdir = a.ui if a.ui else tempfile.mkdtemp(prefix="darkbat_eval_")
    if a.ui:
        os.makedirs(a.ui, exist_ok=True)
    run_dir = link_out = None

    if do_detect:
        run_dir = a.run_dir or build_fixture_run(a.ui or os.path.join(workdir, "out_fake"), a.sar)
        metrics["detection"] = m = score_detection(run_dir, a.truth or (f"fixtures/{a.sar}/truth.csv" if a.sar else "fixtures/s2/truth.csv"))
        print("\n=== DETECTION ===")
        print(f"  truth vessels      {m['truth_vessels']}   detections {m['detections']}")
        print(f"  recall             {m['recall']:.3f}  ({m['tp']} found, {m['fn']} missed)")
        print(f"  precision          {m['precision']:.3f}  ({m['fp']} false alarms, {m['masked_leaks']} masked-vessel leaks)")
        print(f"  length MAE         {m['length_mae_m']:.1f} m")
        print(f"  status accuracy    {m['status_accuracy']:.3f}")
        print(f"  per mode           {m['per_mode']}")
        for k in ("detect_recall", "detect_precision", "status_accuracy"):
            val = m[{"detect_recall": "recall", "detect_precision": "precision", "status_accuracy": "status_accuracy"}[k]]
            if val < THRESHOLDS[k]:
                failures.append(f"{k} {val:.3f} < {THRESHOLDS[k]}")

    if do_link:
        out = link_out = build_link_demo(workdir)
        metrics["linking"] = m = score_linking(out)
        print("\n=== CROSS-PASS LINKING ===")
        print(f"  contacts           {m['contacts']}   tracks {m['tracks']}")
        print(f"  pairwise precision {m['pair_precision']:.3f}")
        print(f"  pairwise recall    {m['pair_recall']:.3f}")
        print(f"  F1                 {m['link_f1']:.3f}")
        if m["link_f1"] < THRESHOLDS["link_f1"]:
            failures.append(f"link_f1 {m['link_f1']:.3f} < {THRESHOLDS['link_f1']}")

    if a.json:
        json.dump(metrics, open(a.json, "w"), indent=2)
        print(f"\nmetrics -> {a.json}")

    if a.ui:
        target = run_dir or link_out
        html = render_ui(target, metrics, failures)
        print(f"\nUI -> {html}  (open it in a browser; see the Evaluation tab)")

    print("\n" + ("FAIL: " + "; ".join(failures) if failures else "PASS: all metrics above thresholds"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
