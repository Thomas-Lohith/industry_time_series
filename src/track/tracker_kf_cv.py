
import os
import sys
import ast
import math
import argparse
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
class CFG:
    # --- model constants (from spec)
    SIGMA_T      = 0.05
    P_D          = 0.95
    T_MERGE      = 0.10
    LAMBDA_FALSE = 0.2
    V_MIN        = 11.0 #m/s -->39.6km/hr physical gate min
    V_MAX        = 55.0 #m/s -->180km/hr physical gate max
    MU_V         = 22.5 # SPEED prior mean -->81km/hr
    SIGMA_V      = 2.0

    # ---tunable (seeding / gating) 
    EPSILON_T    = 0.800
    N_NEIGHBOURS = 4
    DELTA_V      = 2.0
    K_GATE       = 4.0
    K_PRUNE      = 200.0
    N_SCAN       = 3

    # --- PROVISIONAL(tunable) (Kalman filter, CV model) ---
    Q_T              = 1e-6   # process noise on t_last (kept tiny; t is
                               # otherwise deterministic given u and dx)
    Q_U_PER_M        = 0.01   # (m/s)^2 of speed-uncertainty growth per metre
                               # of bridge travelled between hits
    SIGMA_U0_DEFAULT = 2.0    # fallback prior std-dev on a seed's speed
                               # when the seed's cluster has only 1 member(so sample variance can't be computed)


# ============================================================
# STAGE 1 — LOADER  
# ============================================================
def load_detections(paths):
    frames = [pd.read_csv(p) for p in paths]
    det = pd.concat(frames, ignore_index=True)
    det = det.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    det["did"] = np.arange(len(det))
    return det


def build_geometry(det):
    positions = np.sort(det["longitudinal_position"].unique())
    pos_index = {p: i for i, p in enumerate(positions)}
    colocated = {}
    for p, grp in det.groupby("longitudinal_position"):
        colocated[p] = sorted(grp["sensor_id"].unique().tolist())
    return positions, pos_index, colocated


def per_position_times(det, positions):
    idx = {}
    for p in positions:
        sub = det[det["longitudinal_position"] == p].sort_values("detection_time")
        idx[p] = {"t": sub["detection_time"].to_numpy(), "did": sub["did"].to_numpy()}
    return idx


# ============================================================
# STAGE 2 — PHYSICAL GATE + SEEDING  
# ============================================================
def physical_gate_partners(t_k, dx, pos_times_j):
    lo = t_k - dx / CFG.V_MIN - CFG.EPSILON_T
    hi = t_k - dx / CFG.V_MAX + CFG.EPSILON_T
    t = pos_times_j["t"]
    a = np.searchsorted(t, lo, side="left")
    b = np.searchsorted(t, hi, side="right")
    return pos_times_j["did"][a:b], t[a:b]


def generate_seeds(det, positions, pos_index, pos_times):
    seeds = []
    for i, x_k in enumerate(positions):
        down = pos_times[x_k]
        for t_k, did_k in zip(down["t"], down["did"]):
            cand = []
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
            clusters = cluster_velocities(cand, CFG.DELTA_V)
            for cl in clusters:
                vs = [c[0] for c in cl]
                u0 = float(np.mean(vs))
                if abs(u0 - CFG.MU_V) > 4 * CFG.SIGMA_V:
                    continue
                t0 = t_k - (x_k) / u0
                seeds.append({
                    "dids": [int(did_k)] + [c[1] for c in cl],
                    "u0": u0,
                    "u_var": float(np.var(vs, ddof=1)) if len(vs) > 1 else None,
                    "x0": 0.0,
                    "t0": t0,
                    "anchor_did": int(did_k),
                    "anchor_x": float(x_k),
                    "anchor_t": float(t_k),
                })
    return seeds


def cluster_velocities(cand, delta_v):
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
    return clusters


# ============================================================
# STAGE 4/5 — KALMAN FILTER (CV) + TRACK EXTENSION
# ============================================================
@dataclass
class NodeKF:
    dids: list
    state: np.ndarray       # [t_last, u]
    P: np.ndarray           # 2x2 covariance
    x_last: float
    logscore: float
    positions_used: set
    fork_depth: int = 0

    @property
    def u(self):
        return float(self.state[1])

    @property
    def a(self):
        """Time at x=0, under the CV
        assumption. Derived for compatibility with jpda_soft/scoring, which
        expect a (a, u) line — carries no explicit uncertainty of its own."""
        u = self.state[1]
        if u == 0:
            return float("nan")
        return float(self.state[0] - self.x_last / u)


def predict_t(a, u, x):
    """Kept for compatibility with jpda_soft, which predicts at arbitrary
    positions from the track's (a, u) line."""
    return a + x / u


def speed_prior_loglik(u):
    z = (u - CFG.MU_V) / CFG.SIGMA_V
    return -0.5 * z * z - math.log(CFG.SIGMA_V * math.sqrt(2 * math.pi))


def measurement_loglik_kf(innovation, S):
    """Like the original measurement_loglik, but using the filter's own
    predicted innovation variance S instead of the fixed SIGMA_T — this is
    what lets a prediction far from the last hit be scored as less certain."""
    return -0.5 * innovation * innovation / S - 0.5 * math.log(2 * math.pi * S)


def kf_init_cv(seed):
    t_last = seed["anchor_t"]
    u = seed["u0"]
    var_u = seed["u_var"] if seed["u_var"] is not None else CFG.SIGMA_U0_DEFAULT ** 2
    var_u = max(var_u, 0.25)   # floor: don't let P start unrealistically tight
    P0 = np.array([[CFG.SIGMA_T ** 2, 0.0],
                   [0.0, var_u]])
    return np.array([t_last, u], dtype=float), P0, seed["anchor_x"]


def kf_predict_cv(state, P, dx):
    """EKF predict step for t = t_last + dx/u (u assumed constant + process
    noise). Returns None if u has become non-positive (degenerate)."""
    t, u = state
    if u <= 0.1:
        return None
    t_pred = t + dx / u
    u_pred = u
    F = np.array([[1.0, -dx / (u * u)],
                  [0.0, 1.0]])
    Q = np.array([[CFG.Q_T, 0.0],
                  [0.0, CFG.Q_U_PER_M * abs(dx)]])
    P_pred = F @ P @ F.T + Q
    return np.array([t_pred, u_pred]), P_pred


def kf_update_cv(state_pred, P_pred, t_obs):
    H = np.array([1.0, 0.0])
    R = CFG.SIGMA_T ** 2
    S = float(H @ P_pred @ H.T + R)
    K = (P_pred @ H.T) / S
    innovation = t_obs - state_pred[0]
    state_new = state_pred + K * innovation
    P_new = P_pred - np.outer(K, H @ P_pred)
    return state_new, P_new, S, innovation


def extend_track_kf_cv(seed, det, positions, pos_index, pos_times):
    """Greedy best-fit attachment, same branching policy as the original
    extend_track (attach the single best-fitting detection or coast; leave
    contention between overlapping tracks to JPDA) — but the state update at
    each step is now a Kalman predict+update instead of a batch re-fit."""
    did_x = det.set_index("did")["longitudinal_position"].to_dict()

    state, P, x_last = kf_init_cv(seed)
    dids = list(seed["dids"])
    used_pos = set(did_x[d] for d in dids)
    logscore = speed_prior_loglik(state[1])

    for x_s in positions:
        if x_s in used_pos or x_s <= x_last:
            continue
        dx = x_s - x_last
        pred = kf_predict_cv(state, P, dx)
        if pred is None:
            break
        state_pred, P_pred = pred
        t_hat = state_pred[0]
        S = float(P_pred[0, 0] + CFG.SIGMA_T ** 2)
        gate = CFG.K_GATE * math.sqrt(S)

        pt = pos_times[x_s]
        ai = np.searchsorted(pt["t"], t_hat - gate, "left")
        bi = np.searchsorted(pt["t"], t_hat + gate, "right")
        surv = list(zip(pt["did"][ai:bi], pt["t"][ai:bi]))

        if not surv:
            logscore += math.log(1 - CFG.P_D)
            state, P, x_last = state_pred, P_pred, x_s   # coast forward
            continue

        best_did, best_t = min(surv, key=lambda s: abs(s[1] - t_hat))
        used_pos.add(x_s)
        dids.append(int(best_did))
        for did_s, t_s in surv:
            if int(did_s) != int(best_did) and abs(t_s - best_t) <= CFG.EPSILON_T:
                dids.append(int(did_s))

        state, P, S_upd, innovation = kf_update_cv(state_pred, P_pred, best_t)
        x_last = x_s
        logscore += measurement_loglik_kf(innovation, S_upd) + math.log(CFG.P_D)

    logscore += speed_prior_loglik(state[1])
    return [NodeKF(dids, state, P, x_last, logscore, used_pos)]


# ============================================================
# STAGE 6 — JPDA (unchanged in structure; operates on NodeKF via .a/.u)
# ============================================================
def build_tracks(seeds, det, positions, pos_index, pos_times):
    tracks = []
    seen = set()
    for seed in seeds:
        key = tuple(sorted(seed["dids"]))
        if key in seen:
            continue
        seen.add(key)
        best = extend_track_kf_cv(seed, det, positions, pos_index, pos_times)
        if best:
            tracks.append(best[0])
    return dedup_tracks(tracks)


def dedup_tracks(tracks):
    tracks.sort(key=lambda n: (len(n.dids), n.logscore), reverse=True)
    kept = []
    for t in tracks:
        ds = set(t.dids)
        dup = False
        for k in kept:
            ks = set(k.dids)
            inter = len(ds & ks)
            if inter / max(1, min(len(ds), len(ks))) > 0.8:
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


def contested_clusters(tracks):
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
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    soft = {}
    clusters = contested_clusters(tracks)

    for cl in clusters:
        cl_dids = set()
        for ti in cl:
            cl_dids |= set(tracks[ti].dids)

        w = {}
        for ti in cl:
            t = tracks[ti]
            for d in t.dids:
                x = did_x[d]
                t_hat = predict_t(t.a, t.u, x)
                t_obs = det.set_index("did")["detection_time"].to_dict()[d]
                w[(d, ti)] = math.exp(-0.5 * ((t_obs - t_hat) / CFG.SIGMA_T) ** 2)

        claim = {d: [ti for ti in cl if (d, ti) in w] for d in cl_dids}
        soft.update(enumerate_cluster(cl_dids, claim, w, tracks, did_x, det))
    return soft, clusters


def merge_allowed(ti, tj, d, tracks, did_x):
    x = did_x[d]
    thi = predict_t(tracks[ti].a, tracks[ti].u, x)
    thj = predict_t(tracks[tj].a, tracks[tj].u, x)
    return abs(thi - thj) <= CFG.T_MERGE


def enumerate_cluster(cl_dids, claim, w, tracks, did_x, det):
    dids = sorted(cl_dids)
    options = {}
    for d in dids:
        opts = [("t", ti) for ti in claim[d]]
        opts.append(("clutter", None))
        for ti, tj in itertools.combinations(claim[d], 2):
            if merge_allowed(ti, tj, d, tracks, did_x):
                opts.append(("merge", (ti, tj)))
        options[d] = opts

    keys = dids
    total = 1
    for d in keys:
        total *= len(options[d])
    marg = {d: {} for d in keys}
    Z = 0.0

    if total > 200000:
        for d in keys:
            num = {}
            for kind, val in options[d]:
                if kind == "t":
                    num[val] = num.get(val, 0) + w[(d, val)]
                elif kind == "clutter":
                    num["clutter"] = num.get("clutter", 0) + CFG.LAMBDA_FALSE
                else:
                    for ti in val:
                        num[ti] = num.get(ti, 0) + w[(d, ti)]
            s = sum(num.values()) or 1.0
            marg[d] = {k: v / s for k, v in num.items()}
        return marg

    for combo in itertools.product(*[options[d] for d in keys]):
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
# SCORING  
# ============================================================
def _match_tracks_to_vehicles(tracks, det, gt):
    eid = det.set_index("did")["event_id"].to_dict()
    true_map = {}
    for _, row in gt.iterrows():
        for e in ast.literal_eval(row["caused_detection_ids"]):
            true_map.setdefault(e, []).append(int(row["vehicle_id"]))
    match = {}
    for ti, t in enumerate(tracks):
        counts = {}
        for d in t.dids:
            for v in true_map.get(eid[d], []):
                counts[v] = counts.get(v, 0) + 1
        match[ti] = max(counts, key=counts.get) if counts else None
    return match, true_map


def compute_metrics(tracks, soft, det, gt):
    eid = det.set_index("did")["event_id"].to_dict()
    match, true_map = _match_tracks_to_vehicles(tracks, det, gt)

    n_true = len(gt)
    n_est = len(tracks)
    count_err = n_est - n_true

    true_by_vehicle = {int(r["vehicle_id"]): set(ast.literal_eval(r["caused_detection_ids"]))
                       for _, r in gt.iterrows()}
    tp = fp = 0
    total_true_memberships = sum(len(s) for s in true_by_vehicle.values())
    matched_recall_hits = 0
    for ti, t in enumerate(tracks):
        v = match[ti]
        claimed = set(eid[d] for d in t.dids)
        if v is None:
            fp += len(claimed)
            continue
        truth = true_by_vehicle.get(v, set())
        tp += len(claimed & truth)
        fp += len(claimed - truth)
    veh_to_track = {}
    for ti, v in match.items():
        if v is not None:
            veh_to_track.setdefault(v, []).append(ti)
    for v, truth in true_by_vehicle.items():
        captured = set()
        for ti in veh_to_track.get(v, []):
            captured |= set(eid[d] for d in tracks[ti].dids)
        matched_recall_hits += len(truth & captured)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = matched_recall_hits / total_true_memberships if total_true_memberships else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    speed_abs_err = []
    for ti, t in enumerate(tracks):
        v = match[ti]
        if v is None:
            continue
        true_u = float(gt.loc[gt["vehicle_id"] == v, "speed"].iloc[0])
        speed_abs_err.append(abs(t.u - true_u))
    speed_mae = float(np.mean(speed_abs_err)) if speed_abs_err else float("nan")
    speed_rmse = float(np.sqrt(np.mean(np.square(speed_abs_err)))) if speed_abs_err else float("nan")

    vehicles_found = len(set(v for v in match.values() if v is not None))
    vehicles_missed = n_true - vehicles_found
    spurious_tracks = sum(1 for v in match.values() if v is None)
    duplicate_tracks = sum(len(v) - 1 for v in veh_to_track.values() if len(v) > 1)

    metrics = dict(
        n_true=n_true, n_est=n_est, count_err=count_err,
        precision=precision, recall=recall, f1=f1,
        speed_mae_mps=speed_mae, speed_mae_kmh=speed_mae * 3.6,
        speed_rmse_mps=speed_rmse, speed_rmse_kmh=speed_rmse * 3.6,
        vehicles_found=vehicles_found, vehicles_missed=vehicles_missed,
        spurious_tracks=spurious_tracks, duplicate_tracks=duplicate_tracks,
    )

    print("\n=== METRICS (Kalman CV) ===")
    print(f"  vehicles (true / estimated):   {n_true} / {n_est}   (count error {count_err:+d})")
    print(f"  vehicles found / missed:       {vehicles_found} / {vehicles_missed}")
    print(f"  spurious tracks (no vehicle):  {spurious_tracks}")
    print(f"  duplicate tracks (same veh):   {duplicate_tracks}")
    # print(f"  detection precision:           {precision:.3f}")
    # print(f"  detection recall:              {recall:.3f}")
    # print(f"  detection F1:                  {f1:.3f}")
    # print(f"  speed MAE:                     {speed_mae:.2f} m/s  ({speed_mae*3.6:.2f} km/h)")
    # print(f"  speed RMSE:                    {speed_rmse:.2f} m/s  ({speed_rmse*3.6:.2f} km/h)")
    return metrics


def score(tracks, soft, det, gt_path):
    gt = pd.read_csv(gt_path)
    n_tracks = len(tracks)
    n_true = len(gt)
    print(f"\n=== RESULTS (Kalman CV) ===")
    print(f"true vehicles: {n_true}   estimated tracks: {n_tracks}")
    est_kmh = [round(float(t.u) * 3.6, 1) for t in tracks]
    true_kmh = [round(float(s) * 3.6, 1) for s in gt["speed"].tolist()]
    print(f"\ntrue speeds km/h :  {true_kmh}")
    print(f"est  speeds km/h :  {est_kmh}")
    compute_metrics(tracks, soft, det, gt)


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Bridge vehicle tracker — Kalman filter (CV) variant.")
    p.add_argument("--detections", required=True)
    p.add_argument("--gt", "--ground-truth", dest="gt", default=None)
    p.add_argument("--outdir", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.detections):
        sys.exit(f"detections file not found: {args.detections}")
    if args.gt and not os.path.isfile(args.gt):
        sys.exit(f"ground-truth file not found: {args.gt}")
    os.makedirs(args.outdir, exist_ok=True)

    det = load_detections([args.detections])
    positions, pos_index, colocated = build_geometry(det)
    pos_times = per_position_times(det, positions)
    print(f"loaded {len(det)} detections, {len(positions)} positions, "
          f"{det['sensor_id'].nunique()} sensors")

    seeds = generate_seeds(det, positions, pos_index, pos_times)
    print(f"generated {len(seeds)} raw seeds")
    #print(seeds)

    tracks = build_tracks(seeds, det, positions, pos_index, pos_times)
    print(f"built {len(tracks)} tracks after dedup (Kalman CV)")
    #print (tracks)

    soft, clusters = jpda_soft(tracks, det, positions)
    #print(soft)
    print(f"JPDA: {len(clusters)} contested clusters, {len(soft)} detections with soft assignments")

    if args.gt:
        score(tracks, soft, det, args.gt)


if __name__ == "__main__":
    main()