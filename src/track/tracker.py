"""
Bridge Vehicle Tracker — end-to-end implementation of tracker_design.md
Timestamp-only multi-target tracking: gating -> seeding -> MHT -> JPDA.

Run:  python tracker.py <detections.csv> [ground_truth.csv]

All PROVISIONAL parameters are in the CONFIG block. Values that are model
constants (from the spec) are marked accordingly. This first version aims to
RUN and produce inspectable output; it is not yet tuned.
"""

import sys
import ast
import json
import math
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
class CFG:
    # --- model constants (from spec) ---
    SIGMA_T      = 0.05     # timestamp noise std (s)
    P_D          = 0.95     # per-sensor detection prob
    T_MERGE      = 0.10     # merge window (s) -- R3 trigger
    LAMBDA_FALSE = 0.2      # clutter rate /sensor/min
    V_MIN        = 5.0      # m/s  physical gate floor
    V_MAX        = 50.0     # m/s  physical gate ceiling
    MU_V         = 16.7     # speed prior mean
    SIGMA_V      = 2.0      # speed prior std

    # --- PROVISIONAL (tune empirically) ---
    EPSILON_T    = 0.500    # physical-gate time slack (s)  [widened from 50ms]
    N_NEIGHBOURS = 4        # forward neighbour positions used for seeding
    DELTA_V      = 2.0      # seed-cluster tightness (m/s) -- tight, vehicle sep.
    K_GATE       = 4.0      # statistical-gate width multiplier (k * sigma_t)
    K_PRUNE      = 100.0    # MHT relative pruning constant
    N_SCAN       = 3        # N-scan commit horizon (positions)


# ============================================================
# STAGE 1 — LOADER
# ============================================================
def load_detections(paths):
    """Concatenate windows, sort by time, dedup, build per-position index."""
    frames = [pd.read_csv(p) for p in paths]
    det = pd.concat(frames, ignore_index=True)
    det = det.sort_values("detection_time").reset_index(drop=True)
    # dedup exact seam duplicates (same event_id appearing twice)
    det = det.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    det["did"] = np.arange(len(det))           # internal contiguous id
    return det


def build_geometry(det):
    """position <-> sensors, and ordered list of distinct positions."""
    positions = np.sort(det["longitudinal_position"].unique())
    pos_index = {p: i for i, p in enumerate(positions)}
    # co-location: sensors sharing a position
    colocated = {}
    for p, grp in det.groupby("longitudinal_position"):
        colocated[p] = sorted(grp["sensor_id"].unique().tolist())
    return positions, pos_index, colocated


def per_position_times(det, positions):
    """For each position: arrays of (time, did) sorted by time (for searchsorted)."""
    idx = {}
    for p in positions:
        sub = det[det["longitudinal_position"] == p].sort_values("detection_time")
        idx[p] = {
            "t": sub["detection_time"].to_numpy(),
            "did": sub["did"].to_numpy(),
        }
    return idx


# ============================================================
# STAGE 2 — PHYSICAL GATE + SEEDING
# ============================================================
def physical_gate_partners(t_k, dx, pos_times_j):
    """Return dids at upstream position j that satisfy the physical gate for a
    downstream detection at time t_k with baseline dx = x_k - x_j > 0."""
    # dx/vmax - eps <= (t_k - t_j) <= dx/vmin + eps
    #  => t_k - dx/vmin - eps <= t_j <= t_k - dx/vmax + eps
    lo = t_k - dx / CFG.V_MIN - CFG.EPSILON_T
    hi = t_k - dx / CFG.V_MAX + CFG.EPSILON_T
    t = pos_times_j["t"]
    a = np.searchsorted(t, lo, side="left")
    b = np.searchsorted(t, hi, side="right")
    return pos_times_j["did"][a:b], t[a:b]


def generate_seeds(det, positions, pos_index, pos_times):
    """Filter-then-average seeding over N forward neighbours.
    Returns list of seed dicts: {dids:[...], u0:float, t0:float, x0:float}."""
    seeds = []
    npos = len(positions)
    det_t = det.set_index("did")["detection_time"].to_dict()

    for i, x_k in enumerate(positions):
        # downstream detections at this position
        down = pos_times[x_k]
        for t_k, did_k in zip(down["t"], down["did"]):
            # gather candidate velocities to the N forward (upstream in x) neighbours
            # "forward" = smaller x already passed; we look at positions BEHIND x_k
            cand = []  # (v, did_j, x_j)
            for j in range(max(0, i - CFG.N_NEIGHBOURS), i):
                x_j = positions[j]
                dx = x_k - x_j
                if dx <= 0:
                    continue
                dids_j, ts_j = physical_gate_partners(t_k, dx, pos_times[x_j])
                for did_j, t_j in zip(dids_j, ts_j):
                    dt = t_k - t_j
                    if dt <= 0:
                        continue
                    v = dx / dt
                    if CFG.V_MIN <= v <= CFG.V_MAX:
                        cand.append((v, int(did_j), x_j))
            if not cand:
                continue
            # cluster candidate velocities; keep tight clusters
            clusters = cluster_velocities(cand, CFG.DELTA_V)
            for cl in clusters:
                vs = [c[0] for c in cl]
                u0 = float(np.mean(vs))
                # reject seeds whose speed is implausible under the traffic prior
                # (kills short-baseline false pairings at ~6 m/s or ~40 m/s).
                # PROVISIONAL: keep within MU_V +/- 4*SIGMA_V.
                if abs(u0 - CFG.MU_V) > 4 * CFG.SIGMA_V:
                    continue
                # initial line through this detection: t = t0 + (x - x0)/u
                t0 = t_k - (x_k) / u0   # entry_time referenced to x=0
                seeds.append({
                    "dids": [int(did_k)] + [c[1] for c in cl],
                    "u0": u0,
                    "x0": 0.0,
                    "t0": t0,
                    "anchor_did": int(did_k),
                    "anchor_x": float(x_k),
                    "anchor_t": float(t_k),
                })
    return seeds


def cluster_velocities(cand, delta_v):
    """Greedy 1-D clustering by velocity; returns list of clusters (each a list
    of (v,did,x)). A cluster is 'tight' if members are within delta_v."""
    cand = sorted(cand, key=lambda c: c[0])
    clusters = []
    cur = [cand[0]]
    for c in cand[1:]:
        if c[0] - cur[-1][0] <= delta_v:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
    clusters.append(cur)
    # only keep clusters with >=1 member (all qualify; single-member = weak seed)
    return clusters


# ============================================================
# STAGE 4/5 — LINE FIT + MHT TREE
# ============================================================
def fit_line(points):
    """Weighted (here unweighted) least-squares fit of t = t0 + x/u given
    (x, t) points. Returns (t0_at_x0=0, u). Solves t = a + b*x, u = 1/b."""
    xs = np.array([p[0] for p in points], float)
    ts = np.array([p[1] for p in points], float)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return None
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    a, b = coef  # t = a + b x  ; b = 1/u
    if b <= 0:
        return None
    u = 1.0 / b
    return a, u  # a = t at x=0 (entry time ref), u = speed


@dataclass
class Node:
    dids: list                      # detections in this track-so-far
    a: float                        # fitted t at x=0
    u: float                        # fitted speed
    logscore: float
    positions_used: set
    fork_depth: int = 0             # positions since last fork (for N-scan)


def predict_t(a, u, x):
    return a + x / u


def measurement_loglik(t_obs, t_hat):
    z = (t_obs - t_hat) / CFG.SIGMA_T
    return -0.5 * z * z - math.log(CFG.SIGMA_T * math.sqrt(2 * math.pi))


def speed_prior_loglik(u):
    z = (u - CFG.MU_V) / CFG.SIGMA_V
    return -0.5 * z * z - math.log(CFG.SIGMA_V * math.sqrt(2 * math.pi))


def extend_track(seed, det, positions, pos_index, pos_times):
    """Grow one seed into ONE track by greedy best-fit attachment with re-fit.

    Design note: the MHT tree must NOT branch on every gated detection -- with a
    dense, similar-speed vehicle population, many detections fall in each gate at
    every position, so branch-everything explodes to tens of thousands of live
    hypotheses that pruning cannot cut (the score gap between the true and a
    gated-neighbour detection is smaller than log K). That local ambiguity is
    exactly what JPDA resolves GLOBALLY per contested cluster. So track FORMATION
    here is bounded: attach the single best-fitting detection (or skip) at each
    position, re-fitting as we go. Contention between overlapping tracks is left
    for JPDA. Returns [Node]."""
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()

    pts = [(did_x[d], did_t[d]) for d in seed["dids"]]
    fit = fit_line(pts)
    a, u = fit if fit else (seed["t0"], seed["u0"])

    dids = list(seed["dids"])
    used_pos = set(did_x[d] for d in dids)
    logscore = speed_prior_loglik(u)

    for x_s in positions:
        if x_s in used_pos:
            continue
        t_hat = predict_t(a, u, x_s)
        pt = pos_times[x_s]
        lo = t_hat - CFG.K_GATE * CFG.SIGMA_T
        hi = t_hat + CFG.K_GATE * CFG.SIGMA_T
        ai = np.searchsorted(pt["t"], lo, "left")
        bi = np.searchsorted(pt["t"], hi, "right")
        surv = list(zip(pt["did"][ai:bi], pt["t"][ai:bi]))

        if not surv:
            logscore += math.log(1 - CFG.P_D)          # miss
            continue

        # pick the best-fitting survivor, then also attach its co-located twin
        # (co-located validation: both halves of a crossing belong to the track)
        best_did, best_t = min(surv, key=lambda s: abs(s[1] - t_hat))
        used_pos.add(x_s)
        dids.append(int(best_did))
        for did_s, t_s in surv:
            if int(did_s) != int(best_did) and abs(t_s - best_t) <= CFG.EPSILON_T:
                dids.append(int(did_s))            # co-located twin, same position
        fit = fit_line([(did_x[d], did_t[d]) for d in dids])
        if fit:
            a, u = fit
        logscore += (measurement_loglik(best_t, predict_t(a, u, x_s))
                     + math.log(CFG.P_D))

    logscore += speed_prior_loglik(u)
    return [Node(dids, a, u, logscore, used_pos)]


# ============================================================
# STAGE 6 — JPDA (per contested cluster, R3 conditional exclusivity)
# ============================================================
def build_tracks(seeds, det, positions, pos_index, pos_times):
    tracks = []
    seen = set()
    for seed in seeds:
        key = tuple(sorted(seed["dids"]))
        if key in seen:
            continue
        seen.add(key)
        best = extend_track(seed, det, positions, pos_index, pos_times)
        if best:
            tracks.append(best[0])
    return dedup_tracks(tracks)


def dedup_tracks(tracks):
    """Remove near-duplicate tracks (same vehicle seeded multiple times).
    Keep longer/higher-scoring tracks; drop any track that substantially
    overlaps one already kept. PROVISIONAL overlap threshold 0.3."""
    tracks.sort(key=lambda n: (len(n.dids), n.logscore), reverse=True)
    kept = []
    for t in tracks:
        ds = set(t.dids)
        dup = False
        for k in kept:
            ks = set(k.dids)
            inter = len(ds & ks)
            # overlap relative to the SMALLER track (catches subset duplicates)
            if inter / max(1, min(len(ds), len(ks))) > 0.3:
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


def contested_clusters(tracks):
    """Group tracks that share >=1 detection (connectivity)."""
    parent = list(range(len(tracks)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    sets = [set(t.dids) for t in tracks]
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if sets[i] & sets[j]:
                union(i, j)
    groups = {}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def jpda_soft(tracks, det, positions):
    """Exact joint-event enumeration per cluster with R3 conditional exclusivity.
    Returns soft association: {did: {track_idx: prob, 'clutter': prob}}."""
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    soft = {}
    clusters = contested_clusters(tracks)

    for cl in clusters:
        # detections involved in this cluster
        cl_dids = set()
        for ti in cl:
            cl_dids |= set(tracks[ti].dids)

        # per (did, track) base likelihood weight
        w = {}  # (did, ti) -> weight
        for ti in cl:
            t = tracks[ti]
            for d in t.dids:
                x = did_x[d]
                t_hat = predict_t(t.a, t.u, x)
                # crude: use measurement likelihood as weight
                # (time of this det:)
                t_obs = det.set_index("did")["detection_time"].to_dict()[d]
                w[(d, ti)] = math.exp(measurement_loglik(t_obs, t_hat))

        # enumerate joint events over the cluster's detections
        # each detection -> one of its claiming tracks OR clutter,
        # with R3 merge exception (two tracks share iff within t_merge)
        claim = {d: [ti for ti in cl if (d, ti) in w] for d in cl_dids}
        soft.update(enumerate_cluster(cl_dids, claim, w, tracks, did_x, det))
    return soft, clusters


def merge_allowed(ti, tj, d, tracks, did_x):
    """R3: two tracks may share detection d iff predicted crossings within t_merge."""
    x = did_x[d]
    thi = predict_t(tracks[ti].a, tracks[ti].u, x)
    thj = predict_t(tracks[tj].a, tracks[tj].u, x)
    return abs(thi - thj) <= CFG.T_MERGE


def enumerate_cluster(cl_dids, claim, w, tracks, did_x, det):
    """Enumerate exclusive assignments (+ licensed merges) and marginalise."""
    dids = sorted(cl_dids)
    # options per detection: each claiming track, clutter, and merge-pairs if licensed
    options = {}
    for d in dids:
        opts = [("t", ti) for ti in claim[d]]
        opts.append(("clutter", None))
        # licensed merge options: pairs of claiming tracks within t_merge
        for ti, tj in itertools.combinations(claim[d], 2):
            if merge_allowed(ti, tj, d, tracks, did_x):
                opts.append(("merge", (ti, tj)))
        options[d] = opts

    # enumerate joint events (cap to keep tractable)
    keys = dids
    total = 1
    for d in keys:
        total *= len(options[d])
    marg = {d: {} for d in keys}
    Z = 0.0

    if total > 200000:
        # fallback: per-detection independent normalisation (approx)
        for d in keys:
            num = {}
            for kind, val in options[d]:
                if kind == "t":
                    p = w[(d, val)]
                    num[val] = num.get(val, 0) + p
                elif kind == "clutter":
                    num["clutter"] = num.get("clutter", 0) + CFG.LAMBDA_FALSE
                else:
                    for ti in val:
                        num[ti] = num.get(ti, 0) + w[(d, ti)]
            s = sum(num.values()) or 1.0
            marg[d] = {k: v / s for k, v in num.items()}
        return marg

    for combo in itertools.product(*[options[d] for d in keys]):
        # exclusivity check (a track used by >1 detection is fine across different
        # positions; within a joint event a track may claim multiple detections at
        # different positions -- that's normal for a track. Exclusivity is per
        # detection here.)
        prob = 1.0
        assign = {}
        for d, (kind, val) in zip(keys, combo):
            if kind == "t":
                prob *= w[(d, val)]
                assign[d] = [val]
            elif kind == "clutter":
                prob *= CFG.LAMBDA_FALSE
                assign[d] = ["clutter"]
            else:
                prob *= w[(d, val[0])] * w[(d, val[1])]
                assign[d] = list(val)
        Z += prob
        for d in keys:
            for tgt in assign[d]:
                marg[d][tgt] = marg[d].get(tgt, 0.0) + prob

    if Z > 0:
        for d in keys:
            for tgt in marg[d]:
                marg[d][tgt] /= Z
    return marg


# ============================================================
# SCORING (scorer-only; hard projection derived here)
# ============================================================
def score(tracks, soft, det, gt_path):
    gt = pd.read_csv(gt_path)
    # true detection -> vehicle map
    true_map = {}
    for _, row in gt.iterrows():
        for e in ast.literal_eval(row["caused_detection_ids"]):
            true_map.setdefault(e, []).append(int(row["vehicle_id"]))
    eid = det.set_index("did")["event_id"].to_dict()

    # hard projection: each detection -> argmax track
    hard = {}
    for d, dist in soft.items():
        if not dist:
            continue
        best = max(dist, key=dist.get)
        hard[d] = best

    n_tracks = len(tracks)
    n_true = len(gt)
    print(f"\n=== RESULTS ===")
    print(f"true vehicles: {n_true}   estimated tracks: {n_tracks}")

    # track speed comparison (greedy match by speed)
    est_speeds = sorted([t.u for t in tracks])
    true_speeds = sorted(gt["speed"].tolist())
    print(f"\ntrue speeds (sorted):  {[round(s,1) for s in true_speeds]}")
    print(f"est  speeds (sorted):  {[round(s,1) for s in est_speeds]}")

    # detection assignment precision/recall (loose: did the track set cover truth)
    covered = 0
    total_true_det = 0
    for _, row in gt.iterrows():
        tdids = set(ast.literal_eval(row["caused_detection_ids"]))
        total_true_det += len(tdids)
    print(f"\ntotal true detection-memberships: {total_true_det}")
    print(f"detections with a soft assignment: {len(soft)}")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("usage: python tracker.py <detections.csv> [ground_truth.csv]")
        return
    det_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else None

    det = load_detections([det_path])
    positions, pos_index, colocated = build_geometry(det)
    pos_times = per_position_times(det, positions)

    print(f"loaded {len(det)} detections, {len(positions)} positions, "
          f"{det['sensor_id'].nunique()} sensors")

    seeds = generate_seeds(det, positions, pos_index, pos_times)
    print(f"generated {len(seeds)} raw seeds")

    tracks = build_tracks(seeds, det, positions, pos_index, pos_times)
    print(f"built {len(tracks)} tracks after dedup")

    soft, clusters = jpda_soft(tracks, det, positions)
    print(f"JPDA: {len(clusters)} contested clusters, "
          f"{len(soft)} detections with soft assignments")

    if gt_path:
        score(tracks, soft, det, gt_path)


if __name__ == "__main__":
    main()