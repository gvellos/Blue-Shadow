#!/usr/bin/env python3
"""Cross-pass contact linking for Blue Shadow.

A dark contact in one scene may be the same vessel seen in an earlier pass. We do NOT assign identity; we propose a
*tentative* link with a probability, from physics the analyst can check:

  * implied speed  = great-circle distance / time gap.  Reject if it exceeds a hard maximum; down-weight fast transits.
  * size agreement = ratio of the two estimated lengths.
  * time decay     = a 3-hour gap is far stronger evidence than a 3-day gap (and more operationally useful).

Probabilities are capped well below 1 ("tentative", per the brief). Two link kinds:
  * recurring_contact : an earlier unidentified contact reappears as another unidentified contact.
  * resolved_by_ais   : an earlier dark candidate reappears as an AIS-matched vessel -> a *possible* identity to verify.

Chains of best links are assembled into tracks with a stable TRK-* id, so a contact can be carried across passes.
"""
import math
from datetime import datetime

V_MAX_KN = 45.0      # hard ceiling: no link implies a sustained speed above this
V_SOFT_KN = 18.0     # link probability halves around here; realistic transit cruise
TAU_H = 54.0         # time-gap e-folding (~2 days, matched to Sentinel revisit): a next-pass reacquisition is
                     # meaningful evidence, a week-later coincidence is weak
P_CAP = 0.7          # tentative links never exceed this
P_MIN = 0.12         # below this, do not propose
KN_PER_KMH = 1 / 1.852


def parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _size_factor(la, lb):
    """Length agreement in [0.5, 1.0], or None to reject (estimated lengths differ by more than ~2.2x -> a
    different vessel class). Optical lengths are noisy (wakes inflate them), so the tolerance is generous."""
    if not la or not lb:
        return 0.7                                   # unknown length: neutral, do not kill the link
    ratio = min(la, lb) / max(la, lb)
    if ratio < 0.45:
        return None
    return 0.5 + 0.5 * (ratio - 0.45) / 0.55


def score_pair(a, b):
    """a = earlier contact, b = later contact. Both dicts: t (datetime), lat, lon, length, mmsi(optional).
    Returns a link dict or None."""
    dt_h = (b["t"] - a["t"]).total_seconds() / 3600.0
    if dt_h <= 0:
        return None
    sf = _size_factor(a.get("length"), b.get("length"))
    if sf is None:
        return None
    dist_km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    speed_kn = dist_km * KN_PER_KMH / dt_h
    if speed_kn > V_MAX_KN:
        return None
    feasibility = 1.0 / (1.0 + (speed_kn / V_SOFT_KN) ** 4)
    decay = math.exp(-dt_h / TAU_H)
    p = round(P_CAP * feasibility * sf * decay, 3)
    if p < P_MIN:
        return None
    la, lb = a.get("length"), b.get("length")
    return dict(p=p, dt_hours=round(dt_h, 1), dist_km=round(dist_km, 2), implied_speed_kn=round(speed_kn, 1),
                size_ratio=round(min(la, lb) / max(la, lb), 2) if la and lb else None,
                kind="resolved_by_ais" if b.get("mmsi") else "recurring_contact")


def contact_point(rec):
    """Normalise a pipeline record into the minimal fields the linker needs."""
    ev = rec.get("evidence") or {}
    return dict(id=rec["id"], t=parse_time(rec["time_utc"]), lat=rec["lat"], lon=rec["lon"],
                length=rec.get("est_length_m"), status=rec.get("status"),
                mmsi=ev.get("mmsi") if rec.get("status") == "matched" else None)


def propose_links(records, history, top_k=3):
    """Attach `links` (candidate back-pointers to earlier contacts) and a `track_id` to each current record, in place.

    records : this pass's contacts (pipeline output dicts; must have id, time_utc, lat, lon, est_length_m, status).
    history : earlier-pass contacts as normalised points (see contact_point) carrying a `track_id`.
    Only earlier *unidentified* contacts are candidates to be continued; a later contact may be unidentified
    (recurring) or AIS-matched (tentative resolution). Track continuation is one-to-one: each prior contact is
    continued by at most one current contact (its strongest claimant); weaker claimants keep the link as a
    displayed candidate but start a fresh track. Returns the updated history (current points appended).
    """
    hist_by_id = {h["id"]: h for h in history}
    cur = sorted(records, key=lambda r: parse_time(r["time_utc"]))
    points = [contact_point(r) for r in cur]
    candidates = []                                  # (p, current_index, prior_id, link_dict)
    for i, (rec, b) in enumerate(zip(cur, points)):
        rec["links"] = []
        if rec.get("status") not in ("unidentified", "matched"):
            continue
        for h in history:
            if h.get("status") != "unidentified" or h["t"] >= b["t"]:
                continue
            sc = score_pair(h, b)
            if sc:
                link = dict(to_id=h["id"], to_time=h["t"].isoformat(), to_track=h.get("track_id"),
                            to_scene=h.get("scene"), **sc)
                rec["links"].append(link)
                candidates.append((sc["p"], i, h["id"], link))
        rec["links"].sort(key=lambda l: -l["p"])
        rec["links"] = rec["links"][:top_k]
    # greedy one-to-one: strongest links win; a prior and a current are each used once
    assigned, used_prior = {}, set()
    for p, i, hid, link in sorted(candidates, key=lambda x: -x[0]):
        if i in assigned or hid in used_prior:
            continue
        assigned[i], _ = hid, used_prior.add(hid)
    new_points = []
    for i, (rec, b) in enumerate(zip(cur, points)):
        hid = assigned.get(i)
        b["track_id"] = (hist_by_id[hid].get("track_id") if hid else None) or f"TRK-{b['id']}"
        rec["track_id"] = b["track_id"]
        for l in rec["links"]:
            l["chosen"] = bool(hid) and l["to_id"] == hid
        b["scene"] = rec.get("scene")
        new_points.append(b)
    return history + new_points


def build_tracks(all_records):
    """Group records that already carry a track_id into ordered tracks (for a tracks view / tracks.json)."""
    tracks = {}
    for r in all_records:
        tid = r.get("track_id")
        if not tid:
            continue
        tracks.setdefault(tid, []).append(r)
    out = []
    for tid, recs in tracks.items():
        recs = sorted(recs, key=lambda r: parse_time(r["time_utc"]))
        if len(recs) < 2:
            continue                                  # a track worth showing spans at least two passes
        resolved = next((r for r in recs if r.get("status") == "matched"), None)
        out.append(dict(track_id=tid, n_observations=len(recs),
                        first_seen=recs[0]["time_utc"], last_seen=recs[-1]["time_utc"],
                        observations=[dict(id=r["id"], time_utc=r["time_utc"], lat=r["lat"], lon=r["lon"],
                                           status=r.get("status"), reason_code=r.get("reason_code"),
                                           est_length_m=r.get("est_length_m"),
                                           link_p=(r["links"][0]["p"] if r.get("links") else None)) for r in recs],
                        resolved_mmsi=((resolved.get("evidence") or {}).get("mmsi") if resolved else None),
                        max_link_p=max((r["links"][0]["p"] for r in recs if r.get("links")), default=None)))
    return sorted(out, key=lambda t: -(t["max_link_p"] or 0))
