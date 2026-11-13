"""
Bridge vehicle tracker - timestamp-only, Kalman-filter motion model.

Workflow (one function per stage)
---------------------------------
  CSV -> physical_gating -> velocity_scoring -> mht_tree -> jpda -> write_output

  physical_gating   event pairs at two positions with a plausible crossing speed
  velocity_scoring  Gaussian likelihood of each pair's speed (becomes the root
                    score of that pair's hypothesis tree)
  mht_tree          each pair seeds a tree. A Kalman filter is run along the
                    sensor positions: at every position a node predicts the
                    crossing time, gates detections with an ADAPTIVE gate,
                    branches over competing detections, updates, and
                    adds the innovation log-likelihood to its score.
  jpda              conflicting tracks -> global hypotheses -> P[detection, track]
  write_output      per-vehicle trajectories

Motion model (filter state  s = [t, u, g]  at the track's last position x)
    t : crossing time at x            u : speed (m/s)
    g : drift du/dx (m/s per metre)   -- only when motion = "ca"
  Prediction over a step dx:   u' = u + g*dx ,  t' = t + dx / (u + g*dx/2)
  Measurement: the detection time at x_k (noise sigma_t).
  motion = "cv" keeps g = 0 (constant speed); motion = "ca" lets speed drift.
  g is a distance-drift term (empirical), NOT a physical acceleration in m/s^2.
"""
import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    sigma_t: float = 0.05            # s, timestamp noise
    v_min: float = 18.0              # m/s, physical gate
    v_max: float = 55.0
    mu_v: float = 22.5               # speed prior (global; no WIM)
    sigma_v: float = 2.0
    v_sigmas: float = 4.0            # velocity_scoring keeps |v-mu| <= this*sigma_v
    gate_sigmas: float = 4.0         # gate = gate_sigmas * sqrt(S)  (S = innovation var)
    seed_slack: float = 0.8          # s, slack on the pair gate
    min_baseline: float = 40.0       # m, minimum spacing of a seed pair
    seed_ahead: int = 6              # seed partner within this many positions
    k_prune: float = 15.0            # relative log-score pruning threshold
    branch: int = 3                  # max surviving nodes per position
    min_hits: int = 4                # min positions a track must cover
    max_hyps: int = 200              # safety cap on hypotheses per conflict group
    p_output: float = 0.3         # P(track) needed to report it as a vehicle
    # --- Kalman filter (PROVISIONAL, untuned) ---
    motion: str = "ca"               # "cv" constant speed | "ca" speed drifts with distance
    q_t: float = 1e-6                # process noise on time
    q_u_per_m: float = 0.002         # process noise on speed, per metre travelled
    q_g_per_m: float = 1e-6          # process noise on drift, per metre travelled
    sigma_g0: float = 0.001          # prior std of drift (m/s per m). Looser values (0.01)
                                     # gave non-physical drifts on the test window.


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class Data:
    def __init__(self, df):
        self.df = df
        self.t = df["detection_time"].to_numpy(float)
        self.sensor = df["sensor_id"].to_numpy()
        self.positions = np.sort(df["longitudinal_position"].unique())
        idx = {p: i for i, p in enumerate(self.positions)}
        self.pos = df["longitudinal_position"].map(idx).to_numpy()
        self.x = self.positions[self.pos]
        self.by_pos, self.sensors_at = [], []
        for k in range(len(self.positions)):
            ids = np.flatnonzero(self.pos == k)
            ids = ids[np.argsort(self.t[ids])]
            self.by_pos.append((ids, self.t[ids]))
            self.sensors_at.append(sorted(set(self.sensor[ids])))


def load(path):
    df = pd.read_csv(path).drop_duplicates(subset="event_id").reset_index(drop=True)
    return Data(df)


def log_normal(z, sigma):
    return -0.5 * (z / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


# --------------------------------------------------------------------------
# Kalman filter  (state [t, u, g], measurement = time)
# --------------------------------------------------------------------------
def kf_predict(state, P, dx, cfg):
    """Move the state forward by dx metres. Returns (state, P) or None if the
    predicted speed collapses (degenerate track)."""
    t, u, g = state
    if u <= 0.1:
        return None
    s_avg = u + g * dx / 2.0                    # first-order average speed over the step
    if s_avg <= 0.5:
        return None
    F = np.array([[1.0, -dx / s_avg ** 2, -dx ** 2 / (2.0 * s_avg ** 2)],
                  [0.0, 1.0, dx],
                  [0.0, 0.0, 1.0]])
    q_g = cfg.q_g_per_m * dx if cfg.motion == "ca" else 0.0
    Q = np.diag([cfg.q_t, cfg.q_u_per_m * dx, q_g])
    return np.array([t + dx / s_avg, u + g * dx, g]), F @ P @ F.T + Q


def kf_update(state, P, t_obs, cfg):
    """Fuse one detection time. Returns (state, P, innovation, S)."""
    S = P[0, 0] + cfg.sigma_t ** 2
    K = P[:, 0] / S
    nu = t_obs - state[0]
    return state + K * nu, P - np.outer(K, P[0, :]), nu, S


# --------------------------------------------------------------------------
# Stage 1 - physical gating
# --------------------------------------------------------------------------
def physical_gating(D, cfg):
    """Event pairs (d1, d2, v) at two positions with V_MIN <= speed <= V_MAX."""
    pairs = []
    K = len(D.positions)
    for k1 in range(K):
        ids1, ts1 = D.by_pos[k1]
        for k2 in range(k1 + 1, min(K, k1 + 1 + cfg.seed_ahead)):
            dx = D.positions[k2] - D.positions[k1]
            if dx < cfg.min_baseline:
                continue
            ids2, ts2 = D.by_pos[k2]
            for d1, t1 in zip(ids1, ts1):
                lo = np.searchsorted(ts2, t1 + dx / cfg.v_max - cfg.seed_slack)
                hi = np.searchsorted(ts2, t1 + dx / cfg.v_min + cfg.seed_slack, "right")
                for d2, t2 in zip(ids2[lo:hi], ts2[lo:hi]):
                    if t2 > t1 and cfg.v_min <= dx / (t2 - t1) <= cfg.v_max:
                        pairs.append((int(d1), int(d2), dx / (t2 - t1)))
    return pairs


# --------------------------------------------------------------------------
# Stage 2 - velocity likelihood scoring
# --------------------------------------------------------------------------
def velocity_scoring(pairs, cfg):
    """Gaussian log-likelihood of each pair's speed; drop unlikely pairs.
    Returns [(d1, d2, v, loglik)] sorted best first."""
    scored = []
    for d1, d2, v in pairs:
        if abs(v - cfg.mu_v) <= cfg.v_sigmas * cfg.sigma_v:
            scored.append((d1, d2, v, log_normal(v - cfg.mu_v, cfg.sigma_v)))
    scored.sort(key=lambda s: -s[3])
    return scored


# --------------------------------------------------------------------------
# Stage 3 - MHT hypothesis tree with a Kalman filter on every node
# --------------------------------------------------------------------------
@dataclass
class Node:
    dids: tuple
    state: np.ndarray      # [t, u, g] at x_last
    P: np.ndarray          # 3x3 covariance
    x_last: float
    logscore: float


@dataclass
class Track:
    dids: frozenset
    a: float               # entry time reference (time at x = 0, mean-speed line)
    u: float               # final filtered speed
    drift: float           # final filtered drift (m/s per m); 0 for motion = "cv"
    score: float           # root speed log-lik + sum of innovation log-likelihoods


def init_node(D, d1, d2, v, loglik, cfg):
    """Root of a tree. The seed pair only provides the starting state at d1's
    position: time t1, speed v and its uncertainty (from the pair baseline).
    d1 and d2 are NOT pre-attached; they are re-picked by the gate like any
    other detection (so their co-located partners can join too)."""
    dx = D.positions[D.pos[d2]] - D.positions[D.pos[d1]]
    var_v = (v ** 2 * math.sqrt(2) * cfg.sigma_t / dx) ** 2    # speed var from a 2-point fit
    g_var = cfg.sigma_g0 ** 2 if cfg.motion == "ca" else 0.0
    P0 = np.diag([cfg.sigma_t ** 2, var_v, g_var])
    state = np.array([D.t[d1], v, 0.0])
    return Node((), state, P0, float(D.x[d1]), loglik)


def children(D, node, k, cfg):
    """Continue a node at position k: predict, gate (adaptive), update, score.
    Per sensor the node branches over the gated detections (real competition);
    it skips a sensor only when nothing is in the gate. Skipping an available
    detection is NOT a branch: that is what a conflicting track + JPDA express."""
    pred = kf_predict(node.state, node.P, D.positions[k] - node.x_last, cfg)
    if pred is None:
        return []
    st, P = pred
    gate = cfg.gate_sigmas * math.sqrt(P[0, 0] + cfg.sigma_t ** 2)
    ids, ts = D.by_pos[k]
    lo = np.searchsorted(ts, st[0] - gate, "left")
    hi = np.searchsorted(ts, st[0] + gate, "right")
    per_sensor = []
    for s in D.sensors_at[k]:
        cand = [int(d) for d in ids[lo:hi] if D.sensor[d] == s]
        per_sensor.append(cand if cand else [None])      # branch only on real competition
    out = []
    for combo in itertools.product(*per_sensor):
        hits = sorted((d for d in combo if d is not None), key=lambda d: D.t[d])
        s2, P2, gain = st, P, 0.0
        for d in hits:                              # co-located sensors: dx = 0 updates
            s2, P2, nu, S = kf_update(s2, P2, D.t[d], cfg)
            gain += log_normal(nu, math.sqrt(S))
        out.append(Node(node.dids + tuple(hits), s2, P2, float(D.positions[k]),
                        node.logscore + gain))
    return out


def grow_tree(D, root, k_start, cfg):
    """Run the filter forward over all positions from k_start; keep a small set
    of surviving leaves (several, not one)."""
    nodes = [root]
    for k in range(k_start, len(D.positions)):
        nxt = [c for n in nodes for c in children(D, n, k, cfg)]
        if not nxt:
            break
        nxt.sort(key=lambda n: -n.logscore)
        best = nxt[0].logscore
        nodes = [n for n in nxt if n.logscore >= best - cfg.k_prune][:cfg.branch]
    return nodes


def to_track(D, node):
    """Entry-time reference from the track's own first/last detection."""
    ids = sorted(node.dids, key=lambda d: D.x[d])
    f, l = ids[0], ids[-1]
    dt = D.t[l] - D.t[f]
    u_mean = (D.x[l] - D.x[f]) / dt if dt > 0 and D.x[l] > D.x[f] else float(node.state[1])
    return Track(frozenset(node.dids), float(D.t[f] - D.x[f] / u_mean),
                 float(node.state[1]), float(node.state[2]), node.logscore)


def mht_tree(D, scored_pairs, cfg):
    """One hypothesis tree per seed pair; return all surviving candidate tracks.
    Seeds are processed earliest-position first, so a vehicle's full track is
    built before its later (redundant) seeds, which are then skipped."""
    tracks, member_of = {}, {}
    for d1, d2, v, ll in sorted(scored_pairs, key=lambda s: (D.pos[s[0]], -s[3])):
        if member_of.get(d1, set()) & member_of.get(d2, set()):
            continue                                # pair already explained
        root = init_node(D, d1, d2, v, ll, cfg)
        for node in grow_tree(D, root, D.pos[d1], cfg):
            key = frozenset(node.dids)
            if key in tracks or len({D.pos[d] for d in key}) < cfg.min_hits:
                continue
            if node.logscore > 0:                   # drop poorly fitting tracks
                tracks[key] = to_track(D, node)
                for d in key:
                    member_of.setdefault(d, set()).add(key)
    return list(tracks.values())


# --------------------------------------------------------------------------
# Stage 4 - JPDA global resolution (unchanged)
# --------------------------------------------------------------------------
def conflict_groups(tracks):
    parent = list(range(len(tracks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, tr in enumerate(tracks):
        for d in tr.dids:
            if d in owner:
                parent[find(i)] = find(owner[d])
            else:
                owner[d] = i
    groups = {}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def group_hypotheses(tracks, members, cfg):
    hyps = [(0.0, frozenset(), frozenset())]
    for i in sorted(members, key=lambda i: tracks[i].a):
        tr = tracks[i]
        grown = list(hyps)
        for score, used, ids in hyps:
            if used.isdisjoint(tr.dids):
                grown.append((score + tr.score, used | tr.dids, ids | {i}))
        grown.sort(key=lambda h: -h[0])
        best = grown[0][0]
        hyps = [h for h in grown if h[0] >= best - cfg.k_prune][:cfg.max_hyps]
    return [(s, ids) for s, _, ids in hyps]


def jpda(D, tracks, cfg):
    """Returns P[detection, track], p_clutter[detection], p_track[track], stats."""
    n_det, n_trk = len(D.t), len(tracks)
    P = np.zeros((n_det, n_trk))
    p_track = np.zeros(n_trk)
    stats = []
    for members in conflict_groups(tracks):
        hyps = group_hypotheses(tracks, members, cfg)
        scores = np.array([h[0] for h in hyps])
        w = np.exp(scores - scores.max())
        w /= w.sum()
        for (_, ids), p in zip(hyps, w):
            for i in ids:
                p_track[i] += p
        stats.append((len(members), len(hyps)))
    for i, tr in enumerate(tracks):
        P[list(tr.dids), i] = p_track[i]
    p_clutter = np.clip(1.0 - P.sum(axis=1), 0.0, 1.0)
    return P, p_clutter, p_track, stats


# --------------------------------------------------------------------------
# Stage 5 - output
# --------------------------------------------------------------------------
def write_output(D, tracks, P, p_track, cfg, outdir):
    keep = [i for i in np.argsort([tr.a for tr in tracks]) if p_track[i] >= cfg.p_output]
    rows, vehicles = [], []
    for vid, i in enumerate(keep):
        tr = tracks[i]
        for d in sorted(tr.dids, key=lambda d: D.x[d]):
            rows.append(dict(vehicle_id=vid, position=D.x[d], sensor_id=D.sensor[d],
                             event_id=D.df.at[d, "event_id"], detection_time=D.t[d],
                             p_detection_in_track=P[d, i]))
        vehicles.append(dict(vehicle_id=vid, entry_time=tr.a, speed=tr.u, drift=tr.drift,
                             n_detections=len(tr.dids), p_track=p_track[i]))
    traj = pd.DataFrame(rows)
    traj.to_csv(os.path.join(outdir, "trajectories.csv"), index=False)
    pd.DataFrame(vehicles).to_csv(os.path.join(outdir, "vehicles.csv"), index=False)
    return traj, len(keep)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Timestamp-only MHT + JPDA tracker (Kalman)")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--motion", choices=["cv", "ca"], default=Cfg.motion)
    args = ap.parse_args()
    if not os.path.isfile(args.detections):
        sys.exit(f"detections file not found: {args.detections}")
    os.makedirs(args.outdir, exist_ok=True)
    cfg = Cfg(motion=args.motion)

    D = load(args.detections)
    print(f"loaded {len(D.t)} detections, {len(D.positions)} positions, "
          f"{D.df['sensor_id'].nunique()} sensors | motion model: {cfg.motion}")
    pairs = physical_gating(D, cfg)
    print(f"1 physical gating   : {len(pairs)} event pairs")
    scored = velocity_scoring(pairs, cfg)
    print(f"2 velocity scoring  : {len(scored)} pairs kept")
    tracks = mht_tree(D, scored, cfg)
    print(f"3 MHT tree (Kalman) : {len(tracks)} candidate tracks")
    P, p_clutter, p_track, stats = jpda(D, tracks, cfg)
    print(f"4 JPDA              : {len(stats)} conflict groups "
          f"({sum(1 for m, _ in stats if m > 1)} contested), "
          f"largest = {max(m for m, _ in stats)} tracks, "
          f"max hypotheses = {max(h for _, h in stats)}")
    traj, n = write_output(D, tracks, P, p_track, cfg, args.outdir)
    print(f"5 output            : {n} vehicles, {len(traj)} detections in trajectories")
    print(f"  detections with uncertain origin (5%-95% clutter): "
          f"{int(((p_clutter > 0.05) & (p_clutter < 0.95)).sum())}")






if __name__ == "__main__":
    main()