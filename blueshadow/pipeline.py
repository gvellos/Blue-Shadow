#!/usr/bin/env python3
"""Blue Shadow prototype: SAR detection + AIS fusion + explained unidentified contacts.

Runs end-to-end on a SIMULATED Sentinel-1-like scene (gamma speckle sea, ships of known
length, AIS with gaps, a wind farm, a land strip, a no-AIS-coverage strip).
Swap points for real data are marked  # >>> REAL DATA
"""
import json, math, os
from datetime import datetime, timezone
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("OUT", "runs/demo")
os.makedirs(f"{OUT}/crops", exist_ok=True)

PX = 10.0                      # metres per pixel (S1 IW GRD)
H = W = 1500
T_IMG = datetime(2026, 11, 7, 5, 40, tzinfo=timezone.utc)
LAT0, LON0 = 55.45, 14.10      # scene top-left
LAND_COLS = 150
NOCOV_X = 1300 * PX            # simulated: no AIS receiver coverage east of this
SENSITIVE = (5000, 0, 6500, 15000)   # x0,y0,x1,y1 metres: cable/route corridor
BASE_ERR, FRESH_AGE, BELOW_LIMIT_M = 60.0, 600.0, 20.0
GUARD, BG, K_SEED, K_LOW = 41, 91, 8.0, 5.0
BANDS = [("<15 m", 0, 15), ("15-20 m", 15, 20), ("20-50 m", 20, 50),
         ("50-100 m", 50, 100), (">100 m", 100, 1e9)]


def to_latlon(x, y):
    lat = LAT0 - y / 111320.0
    return lat, LON0 + x / (111320.0 * math.cos(math.radians(LAT0)))


def vec(cog):  # unit vector (east, south) for course clockwise from north
    r = math.radians(cog)
    return np.array([math.sin(r), -math.cos(r)])


# ----------------------------------------------------------------- simulation
def make_clutter(rng):
    mean = 1.0 + 0.2 * ndi.gaussian_filter(rng.normal(size=(H, W)), 80) / 0.02
    mean = np.clip(mean, 0.7, 1.4)
    return (mean * rng.gamma(4, 0.25, (H, W))).astype(np.float32)


def render_ship(img, x, y, L, rng):
    wd = max(L / 6.0, 6.0)
    nl, nw = max(L / PX, 1.0), max(wd / PX, 1.0)
    E = 0.03 * L ** 2 * rng.lognormal(0, 0.4)
    ang = rng.uniform(0, math.pi)
    ca, sa = math.cos(ang), math.sin(ang)
    h = int(nl / 2) + 2
    rr, cc = np.mgrid[-h:h + 1, -h:h + 1]
    u, v = cc * ca + rr * sa, -cc * sa + rr * ca
    m = (np.abs(u) <= nl / 2) & (np.abs(v) <= nw / 2)
    m[h, h] = True
    R, C = int(round(y / PX)) + rr[m], int(round(x / PX)) + cc[m]
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    img[R[ok], C[ok]] += E / m.sum() * rng.gamma(4, 0.25, ok.sum())


def make_world(rng):
    ships = []

    def place(L, mode, xr=None, yr=None):
        xr = xr or ((LAND_COLS + 40) * PX, (W - 40) * PX)
        yr = yr or (40 * PX, (H - 40) * PX)
        for _ in range(2000):
            x, y = rng.uniform(*xr), rng.uniform(*yr)
            if all(math.hypot(x - s["x"], y - s["y"]) > 400 for s in ships):
                break
        if x >= NOCOV_X and mode in ("ok", "gap"):
            mode = "nocov"
        ships.append(dict(id=len(ships), x=x, y=y, L=float(L), sog=rng.uniform(2, 16),
                          cog=rng.uniform(0, 360), mode=mode))

    pick = lambda p: rng.choice(["ok", "gap", "dark"], p=p)
    for _ in range(28): place(np.clip(rng.lognormal(math.log(120), 0.5), 50, 300), pick([.85, .08, .07]))
    for _ in range(22): place(rng.uniform(20, 50), pick([.55, .05, .40]))
    for _ in range(30): place(rng.uniform(6, 20), pick([.25, 0, .75]))
    for _ in range(2): place(rng.uniform(80, 160), "dark", xr=(5100, 6400))   # dark ships in sensitive corridor
    turbines = [(900 * PX + i * 80 * PX, 300 * PX + j * 80 * PX) for i in range(4) for j in range(3)]
    return ships, turbines


def make_ais(rng, ships):  # >>> REAL DATA: load DMA AIS CSV -> same message schema
    msgs = []
    for s in ships:
        if s["mode"] not in ("ok", "gap"):
            continue
        mmsi, v, d = 200000000 + s["id"], s["sog"] * 0.5144, vec(s["cog"])
        p0 = np.array([s["x"], s["y"]])
        if s["mode"] == "ok":
            ts = np.arange(-3600, rng.uniform(300, 900), rng.choice([10, 30, 60, 180, 360]))
            rep_cog = s["cog"]
        else:
            age = rng.uniform(900, 3300)
            ts = np.arange(-age - 3000, -age + 1, rng.choice([30, 60, 180]))
            rep_cog = s["cog"] + rng.choice([-1, 1]) * rng.uniform(5, 25)   # vessel turned after last ping
        for t in ts:
            p = p0 + v * t * d + rng.normal(0, 12, 2)
            msgs.append((mmsi, float(t), p[0], p[1], s["sog"], rep_cog, s["L"]))
    for k in range(12):  # AIS vessels outside the scene
        msgs.append((300000000 + k, 0.0, rng.uniform(16000, 30000), rng.uniform(0, 15000), 8.0, 90.0, 100.0))
    tracks = {}
    for m in msgs:
        tracks.setdefault(m[0], []).append(m)
    return {k: np.array(sorted(v)) for k, v in tracks.items()}


def make_scene(rng):  # >>> REAL DATA: read S1 GRD VV (calibrated sigma0 -> power), geocode, land mask
    img = make_clutter(rng)
    ships, turbines = make_world(rng)
    for s in ships: render_ship(img, s["x"], s["y"], s["L"], rng)
    for (x, y) in turbines:
        img[int(y / PX), int(x / PX)] += 400
    img[:, :LAND_COLS] = 6 * rng.gamma(2, 0.5, (H, LAND_COLS))
    land = np.zeros((H, W), bool); land[:, :LAND_COLS] = True
    return img, land, ships, turbines, make_ais(rng, ships)


# ------------------------------------------------------------------ detection
def detect(img, excl):
    box = lambda a, k: ndi.uniform_filter(a, k, mode="reflect")
    nb, ng = BG ** 2, GUARD ** 2
    n = nb - ng
    mu = (box(img, BG) * nb - box(img, GUARD) * ng) / n
    var = np.maximum((box(img ** 2, BG) * nb - box(img ** 2, GUARD) * ng) / n - mu ** 2, 1e-6)
    z = (img - mu) / np.sqrt(var)
    lab, _ = ndi.label((z > K_LOW) & ~excl, structure=np.ones((3, 3)))
    seeds = np.unique(lab[(z > K_SEED) & ~excl]); seeds = seeds[seeds > 0]
    objs, dets = ndi.find_objects(lab), []
    for l in seeds:
        sl = objs[l - 1]
        rr, cc = np.nonzero(lab[sl] == l)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        w = img[rr, cc]
        lam = np.linalg.eigvalsh(np.cov(np.vstack([rr, cc]))).max() if len(rr) > 1 else 0.0
        pk = np.argmax(w)
        dets.append(dict(x=float((cc * w).sum() / w.sum() * PX), y=float((rr * w).sum() / w.sum() * PX),
                         L_est=float(math.sqrt(12 * lam + 1) * PX), area_px=int(len(rr)),
                         peak_z=float(z[rr[pk], cc[pk]]),
                         snr_db=float(10 * math.log10(w[pk] / max(mu[rr[pk], cc[pk]], 1e-3)))))
    return dets


# ------------------------------------------------------------------------ AIS
def interpolate(tracks, t=0.0):
    out = []
    for mmsi, tr in tracks.items():
        ts = tr[:, 1]
        i = np.searchsorted(ts, t, side="right") - 1
        if i < 0:
            continue
        age, v = t - ts[i], tr[i, 4] * 0.5144
        if i + 1 < len(ts):
            f = age / (ts[i + 1] - ts[i])
            pos = tr[i, 2:4] * (1 - f) + tr[i + 1, 2:4] * f
            span = ts[i + 1] - ts[i]
            radius, method, fresh = BASE_ERR + 0.25 * v * span / 2, "interpolated", span <= 1200
        else:
            pos = tr[i, 2:4] + v * age * vec(tr[i, 5])
            radius, method, fresh = BASE_ERR + 0.3 * v * age, "dead_reckoned", age <= FRESH_AGE
        out.append(dict(mmsi=int(mmsi), x=float(pos[0]), y=float(pos[1]), radius=float(radius), age_s=float(age),
                        method=method, fresh=bool(fresh), length=float(tr[i, 6])))
    return out


def match(dets, ais):
    fresh = [a for a in ais if a["fresh"]]
    if not dets or not fresh:
        return {}
    D = np.array([[math.hypot(d["x"] - a["x"], d["y"] - a["y"]) for a in fresh] for d in dets])
    G = np.array([[a["radius"] + 0.5 * d["L_est"] for a in fresh] for d in dets])
    cost = np.where(D <= G, D / G, 1e6)
    r, c = linear_sum_assignment(cost)
    return {int(i): (fresh[j], float(D[i, j])) for i, j in zip(r, c) if cost[i, j] < 1e5}


# ----------------------------------------------------------- reasons + score
def explain(d, ais, matched_mmsi, turbines, d_land):
    dist = lambda a: math.hypot(d["x"] - a["x"], d["y"] - a["y"])
    for (tx, ty) in turbines:
        if math.hypot(d["x"] - tx, d["y"] - ty) < 150:
            return "STATIC_STRUCTURE", "Within 150 m of catalogued wind-turbine position.", {}
    if d["x"] >= NOCOV_X:
        return "NO_AIS_COVERAGE", "Outside AIS receiver coverage at image time; absence of AIS is uninformative.", {}
    if d_land < 600:
        return "NEAR_COAST_CLUTTER", f"{d_land:.0f} m from land: high clutter / structure risk.", {"dist_land_m": round(d_land)}
    for a in sorted(ais, key=dist):
        if a["mmsi"] in matched_mmsi or a["fresh"]:
            continue
        if dist(a) <= a["radius"] + 0.5 * d["L_est"] and 0.3 <= d["L_est"] / a["length"] <= 3:
            ev = dict(mmsi=a["mmsi"], last_ping_min_before=round(a["age_s"] / 60, 1), dist_to_dead_reckoned_m=round(dist(a)),
                      search_radius_m=round(a["radius"]), ais_length_m=a["length"])
            return "POSSIBLE_AIS_GAP", "Inside drift envelope of a vessel whose AIS went silent; plausible dropout, not proof.", ev
    near = [a for a in ais if a["fresh"] and a["mmsi"] not in matched_mmsi and dist(a) < 1500 and 0.3 <= d["L_est"] / a["length"] <= 3]
    if near:
        a = min(near, key=dist)
        return "AIS_MISMATCH", "Fresh AIS track nearby but outside position gate: possible position error or spoofing.", \
            dict(mmsi=a["mmsi"], dist_m=round(dist(a)), gate_m=round(a["radius"] + 0.5 * d["L_est"]))
    if d["L_est"] <= BELOW_LIMIT_M:
        return "SMALL_BELOW_LIMIT", f"Estimated ~{d['L_est']:.0f} m: at/below reliable Sentinel limit. Could be a boat or clutter.", {}
    return "DARK_CANDIDATE", "Ship-sized, open water, no AIS track able to explain it.", {}


def confidence(d, d_land):
    c_snr = 1 - math.exp(-(d["peak_z"] - 5) / 10)
    c_size = 1.0 if 15 <= d["L_est"] <= 400 else 0.5
    c_clutter = float(np.clip(d_land / 1000, 0.2, 1))
    return round(c_snr ** 0.5 * c_size ** 0.2 * c_clutter ** 0.3, 3), \
        dict(contrast=round(c_snr, 2), size_plausibility=c_size, clutter_distance=round(c_clutter, 2))


PRIORITY = dict(DARK_CANDIDATE=1.0, AIS_MISMATCH=.6, POSSIBLE_AIS_GAP=.5, NO_AIS_COVERAGE=.5,
                NEAR_COAST_CLUTTER=.3, SMALL_BELOW_LIMIT=.2, STATIC_STRUCTURE=0.0)


# -------------------------------------------------------------- detection limit
def injection_curve(rng, excl, lengths, trials=2, n=40):
    res = []
    for L in lengths:
        hit = tot = fa = 0
        for _ in range(trials):
            img, pts = make_clutter(rng), []
            while len(pts) < n:
                p = (rng.uniform((LAND_COLS + 60) * PX, (W - 40) * PX), rng.uniform(40 * PX, (H - 40) * PX))
                if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 500 for q in pts): pts.append(p)
            for p in pts: render_ship(img, p[0], p[1], L, rng)
            dets = detect(img, excl)
            used = 0
            for p in pts:
                if any(math.hypot(p[0] - d["x"], p[1] - d["y"]) < 30 + L / 2 for d in dets): hit += 1
            for d in dets:
                if not any(math.hypot(p[0] - d["x"], p[1] - d["y"]) < 30 + L / 2 for p in pts): fa += 1
            tot += n
        res.append(dict(length_m=L, p_det=hit / tot, n=tot, false_alarms_per_scene=fa / trials))
    return res


# ------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(7)
    img, land, ships, turbines, tracks = make_scene(rng)
    d_land = ndi.distance_transform_edt(~land) * PX
    excl = d_land < 200

    dets = detect(img, excl)
    ais = interpolate(tracks, 0.0)
    ais_scene = [a for a in ais if 0 <= a["x"] < W * PX and 0 <= a["y"] < H * PX]
    pairs = match(dets, ais_scene)
    matched_mmsi = {a["mmsi"] for a, _ in pairs.values()}

    curve = injection_curve(np.random.default_rng(11), excl, [8, 10, 12, 15, 20, 25, 30, 40, 60, 100, 150])
    p_det = lambda L: float(np.interp(L, [c["length_m"] for c in curve], [c["p_det"] for c in curve]))

    records, n_id = [], 0
    for i, d in enumerate(dets):
        n_id += 1
        lat, lon = to_latlon(d["x"], d["y"])
        dl = float(d_land[int(d["y"] / PX), int(d["x"] / PX)])
        conf, brk = confidence(d, dl)
        truth = any(math.hypot(s["x"] - d["x"], s["y"] - d["y"]) < 60 + s["L"] / 2 for s in ships)
        rec = dict(id=f"BS-{T_IMG:%Y%m%d}-{n_id:04d}", lat=round(lat, 5), lon=round(lon, 5), time_utc=T_IMG.isoformat(),
                   confidence=conf, confidence_breakdown=brk, est_length_m=round(d["L_est"]),
                   est_length_uncertainty_m=10 + round(0.3 * d["L_est"]), snr_db=round(d["snr_db"], 1),
                   _x=d["x"], _y=d["y"], _truth_vessel=truth)
        if i in pairs:
            a, dd = pairs[i]
            rec.update(status="matched", reason_code="AIS_MATCH", evidence=dict(mmsi=a["mmsi"], dist_m=round(dd),
                       ais_method=a["method"], ais_age_s=round(a["age_s"])))
        else:
            code, text, ev = explain(d, ais_scene, matched_mmsi, turbines, dl)
            inz = SENSITIVE[0] <= d["x"] <= SENSITIVE[2] and SENSITIVE[1] <= d["y"] <= SENSITIVE[3]
            rec.update(status="static_structure" if code == "STATIC_STRUCTURE" else "unidentified",
                       reason_code=code, reason_text=text, evidence=ev, in_sensitive_zone=inz,
                       priority=round(PRIORITY[code] * conf * (1.5 if inz else 1), 3), links=[], analyst_state="new")
            r0, c0 = int(d["y"] / PX), int(d["x"] / PX)
            crop = np.clip(10 * np.log10(np.maximum(img[max(r0 - 32, 0):r0 + 32, max(c0 - 32, 0):c0 + 32], 1e-3)), -3, 20)
            if rec["status"] == "unidentified":
                plt.imsave(f"{OUT}/crops/{rec['id']}.png", crop, cmap="gray", vmin=-3, vmax=20)
                rec["crop"] = f"crops/{rec['id']}.png"
        records.append(rec)

    queue = sorted([r for r in records if r["status"] == "unidentified"], key=lambda r: -r["priority"])
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}
    json.dump([strip(r) for r in queue], open(f"{OUT}/unidentified_contacts.json", "w"), indent=2)
    json.dump([strip(r) for r in records], open(f"{OUT}/all_detections.json", "w"), indent=2)

    # ---- size-band report (AIS reference + simulation truth)
    rows = []
    for name, lo, hi in BANDS:
        ref = [a for a in ais_scene if a["fresh"] and lo <= a["length"] < hi]
        det = [a for a in ref if a["mmsi"] in matched_mmsi]
        tr = [s for s in ships if lo <= s["L"] < hi]
        trd = [s for s in tr if any(math.hypot(s["x"] - d["x"], s["y"] - d["y"]) < 60 + s["L"] / 2 for d in dets)]
        rows.append(dict(band=name, ais_vessels_in_scene=len(ref), ais_detected=len(det),
                         recall_vs_ais=round(len(det) / len(ref), 2) if ref else None,
                         expected_detectable=round(sum(p_det(a["length"]) for a in ref), 1),
                         sim_truth_vessels=len(tr), sim_truth_detected=len(trd)))
    fa = [r for r in records if r["status"] != "static_structure" and not r["_truth_vessel"]]
    bins, calib = [0, .3, .5, .7, .85, 1.01], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [r for r in records if lo <= r["confidence"] < hi and r["status"] != "static_structure"]
        calib.append(dict(conf_bin=f"{lo:.2f}-{min(hi, 1):.2f}", n=len(b),
                          frac_real=round(sum(r["_truth_vessel"] for r in b) / len(b), 2) if b else None))
    reasons = {}
    for r in queue: reasons[r["reason_code"]] = reasons.get(r["reason_code"], 0) + 1
    report = dict(note="SIMULATED scene. Detection curve reflects the simulator's ship model; recalibrate with real S1 chips.",
                  detections=len(dets), matched=len(pairs), unidentified=len(queue), reason_breakdown=reasons,
                  false_alarms_non_structure=len(fa), size_bands=rows, detection_limit_curve=curve, calibration=calib)
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)

    # ---- figures
    colors = dict(matched="#2ca02c", DARK_CANDIDATE="#d62728", AIS_MISMATCH="#ff7f0e", POSSIBLE_AIS_GAP="#ff7f0e",
                  NO_AIS_COVERAGE="#ff7f0e", NEAR_COAST_CLUTTER="#888", SMALL_BELOW_LIMIT="#888", STATIC_STRUCTURE="#444")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(10 * np.log10(np.maximum(img, 1e-3)), cmap="gray", vmin=-3, vmax=14)
    for r in records:
        c = colors["matched"] if r["status"] == "matched" else colors[r["reason_code"]]
        ax.add_patch(plt.Circle((r["_x"] / PX, r["_y"] / PX), 22, fill=False, ec=c, lw=1.5))
    ax.axvline(NOCOV_X / PX, color="c", ls="--", lw=1); ax.text(NOCOV_X / PX + 5, 30, "no AIS coverage", color="c")
    ax.set_title("Detections: green=AIS matched, red=dark candidate, orange=explainable gap/mismatch, grey=low-trust"); ax.axis("off")
    fig.savefig(f"{OUT}/scene_overview.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    L = [c["length_m"] for c in curve]
    a1.plot(L, [c["p_det"] for c in curve], "o-"); a1.axvspan(0, BELOW_LIMIT_M, color="orange", alpha=.2)
    a1.text(8, .05, "below Sentinel\nreliable limit", fontsize=9); a1.set_xscale("log")
    a1.set(xlabel="vessel length (m)", ylabel="detection probability", title="Measured detection limit (injection, simulated)")
    x = np.arange(len(rows)); a2.bar(x - .2, [r["expected_detectable"] for r in rows], .4, label="expected detectable")
    a2.bar(x + .2, [r["ais_detected"] for r in rows], .4, label="detected (AIS-matched)")
    a2.set_xticks(x); a2.set_xticklabels([r["band"] for r in rows]); a2.legend(); a2.set_title("AIS vessels in scene by size band")
    fig.savefig(f"{OUT}/size_band_and_limit.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    print(f"detections={len(dets)} matched={len(pairs)} unidentified={len(queue)} reasons={reasons}")
    print("band, ais_in_scene, ais_detected, recall, expected_detectable, sim_truth, sim_detected")
    for r in rows: print(r["band"], r["ais_vessels_in_scene"], r["ais_detected"], r["recall_vs_ais"], r["expected_detectable"], r["sim_truth_vessels"], r["sim_truth_detected"])
    print("detection curve:", [(c["length_m"], round(c["p_det"], 2)) for c in curve])
    print("calibration:", calib, "| non-vessel detections:", len(fa))


if __name__ == "__main__":
    main()
