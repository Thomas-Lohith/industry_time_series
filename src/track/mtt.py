#!/usr/bin/env python3
"""
bridge_mtt.py - vehicle tracking on a bridge from extracted accelerometer
detection timestamps.            Usage:  python bridge_mtt.py params.yaml

Vehicles travel towards increasing longitudinal_position.

Motion models (config key `motion`, default "cv"):
  cv  state s = [t, u]       constant speed, speed random walk
        t' = t + dx/u,               u' = u + w_u
  ca  state s = [t, u, g]    speed drifts linearly with distance, g = du/dx
        t' = t + dx/(u + g*dx/2),    u' = u + g*dx + w_u,   g' = g + w_g
      g is a distance drift (m/s per metre), NOT an acceleration in m/s^2
      (the time-domain acceleration would be a = u*g).
  w_u, w_g: white noise on du/dx and dg/dx with intensities q_u, q_g per metre;
  Q is the integrated-noise covariance of the linearised model.
  With g = 0 and q_g = 0 the ca model reduces exactly to cv.

Shared by all methods:
  measure  z = t_det - bias = t + e,       e ~ N(0, sigma_t^2)
  score    log-likelihood ratio, vehicle vs clutter
           birth ln(P_D*birth_rate/lam_fa) + misses before it
           hit   ln P_D + ln N(nu; 0, S) - ln lam_fa
           miss  ln(1 - P_D*P_G)
Methods: mht (track-oriented MHT + MILP), jpda, hybrid (JPDA + MHT in
ambiguous zones, one MILP picks the final set).

Evaluation (when truth is given):
  detections vs ground truth  (once)        -> out/detector_metrics.csv
  tracks vs ground truth      (per method)  -> out/<method>/sensor_metrics.csv
"""
import json
import math
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace


import numpy as np
import pandas as pd
import yaml


#perform sensitivity analysis to these parametrs(MAX_HYPS, MAX_EVENTS, MAX_ZONE_TRACKS, WHICH ARE implementation-level computational limits. 
# They should not be confused with the physical parameters of bridge or vehicle dynamics. 
GATE = 3.0                                   # gate half-width [std devs]#considers a detection only if it falls within three standard deviations of the predicted detection time.
P_GATE = math.erf(GATE / math.sqrt(2.0))     # P(true detection inside gate) #probability that a normally distributed measurement falls within the gate.
MAX_HYPS = 1000                               # safety cap: hypotheses per conflict group #Maximum number of hypotheses maintained in one MHT conflict group.
MAX_EVENTS = 200000                           # safety cap: JPDA joint events per cluster #Maximum number of joint association events enumerated by JPDA.
MAX_ZONE_TRACKS = 12                         # safety cap: hybrid zone size #Maximum number of tracks allowed in one hybrid ambiguous region.
MIN_HITS, MIN_PURITY, MIN_COMPLETE = 4, 0.7, 0.5   # evaluation matching rules
MIN_SPEED = 0.5                              # m/s, below this a track is degenerate
LOG2PI = math.log(2.0 * math.pi)
KEYS = ("detections sensors truth outdir methods sigma_t p_d lam_fa "
        "birth_rate mu_v sigma_v0 q_u beam llr_window min_positions amb_thresh").split()
CA_KEYS = ("q_g", "sigma_g0")                # required only when motion = "ca"


def load_config(path):
    with open(path) as f:
        p = yaml.safe_load(f)
    missing = [k for k in KEYS if k not in p]
    p.setdefault("motion", "cv")
    if p["motion"] not in ("cv", "ca"):
        sys.exit(f"motion must be 'cv' or 'ca', got {p['motion']!r}")
    if p["motion"] == "ca":
        missing += [k for k in CA_KEYS if k not in p]
    if missing:
        sys.exit(f"missing in {path}: {missing}")
    return SimpleNamespace(**p) #converts the dictionary into an object.


# ==========================================================================
# Data
# ==========================================================================
class Data:
    def __init__(self, det, sensors, cfg):
        s = sensors.copy()
        for col, default in (("p_d", cfg.p_d), ("lam_fa", cfg.lam_fa),
                             ("bias", 0.0), ("sigma_t", cfg.sigma_t)):
            s[col] = s[col].fillna(default) if col in s else default
        s = s.sort_values(["longitudinal_position", "sensor_id"]).reset_index(drop=True)
        self.n_scans = len(s)
        self.scan_x = s["longitudinal_position"].to_numpy(float)
        self.scan_sensor = s["sensor_id"].to_numpy()
        self.pd = s["p_d"].to_numpy(float)
        self.lam = s["lam_fa"].to_numpy(float)
        self.sig2 = s["sigma_t"].to_numpy(float) ** 2 #Stores timing variance.--measuremnt variance
        self.scan_pos = np.searchsorted(np.unique(self.scan_x), self.scan_x) #assigns a postion index to every sensor
        self.miss_ll = np.log(1.0 - self.pd * P_GATE) #probability of a detection being missed inside the tracking gate
        self.cum_miss = np.concatenate([[0.0], np.cumsum(self.miss_ll)]) #calculates the cumulative missed-detection penalty.
        j_of = {sid: j for j, sid in enumerate(self.scan_sensor)} #Map sensor IDs to scan indices
        det = det[det["sensor_id"].isin(j_of)]
        self.det = det.drop_duplicates("event_id").reset_index(drop=True) #assiging a vairiable 'det' for event_id in detections.csv
        self.scan = self.det["sensor_id"].map(j_of).to_numpy(int)##assigining scan varibale for sensor_id
        self.z = (self.det["detection_time"].to_numpy(float)#detection time  based on the event_id
                  - s["bias"].to_numpy(float)[self.scan]) #Correct detection timestamps
        self.by_scan = []
        for j in range(self.n_scans):
            ids = np.flatnonzero(self.scan == j)
            ids = ids[np.argsort(self.z[ids])]
            self.by_scan.append((ids, self.z[ids]))


def load(cfg):
    det = pd.read_csv(cfg.detections, dtype={"event_id": str})
    if cfg.sensors:
        sensors = pd.read_csv(cfg.sensors)
    else:
        print("WARNING: no sensor table; sensors without detections are invisible, "
              "so their misses are not scored.", file=sys.stderr)
        sensors = det[["sensor_id", "longitudinal_position"]].drop_duplicates("sensor_id")
    return Data(det, sensors, cfg)


def n_positions(D, dids):
    return len({D.scan_pos[D.scan[d]] for d in dids})
    #calculates the number of unique sensor positions represented by a set of detections.
    #function helps reject tracks that have too few unique sensor positions


# ==========================================================================
# Shared model
# ==========================================================================
def log_npdf(nu, var):
    '''calculates the log of the Gaussian probability density
    When a new detection arrives, the algorithm compares the actual detection time with the predicted detection time.
    If the residual is small, the detection is more likely to belong to that vehicle '''
    return -0.5 * (nu * nu / var + math.log(var) + LOG2PI)


def hit_ll(D, j, nu, S):
    ''' calculates the log-likelihood ratio for associating a detection with an existing vehicle.
    Hypothesis 1: The detection belongs to an existing vehicle.
    Hypothesis 2: The detection is clutter. 
    A higher LLR indicates stronger evidence that the detection belongs to the vehicle.'''
    return math.log(D.pd[j]) + log_npdf(nu, S) - math.log(D.lam[j])



def birth_llr(D, d, cfg):
    '''calculates the likelihood ratio for starting a new vehicle track from a detection.
    Why is cum_miss added?
    Suppose a vehicle enters the bridge before sensor 3.
    If its first detection appears at sensor 3, the function accounts for the possibility that it should have been detected by earlier sensors.
    Therefore, the earlier missed detections reduce the birth score.
    This discourages creating vehicle tracks from detections that occur too late without sufficient evidence '''
    j = D.scan[d] #get the sensor_id  of the detection 'd'
    return math.log(D.pd[j] * cfg.birth_rate / D.lam[j]) + D.cum_miss[j]
    #log(detection probability of that sensor_id * birth_rate: 0.03 / false events per second per sensor + )

def init_state(D, d, cfg):
    """New track from one detection: time from it, speed from the prior,
    drift (ca only) from a zero-mean prior.
    function initializes a new vehicle track using its first detection.."""
    if cfg.motion == "ca":
        return (np.array([D.z[d], cfg.mu_v, 0.0]),
                np.diag([D.sig2[D.scan[d]], cfg.sigma_v0 ** 2, cfg.sigma_g0 ** 2]))
    return np.array([D.z[d], cfg.mu_v]), np.diag([D.sig2[D.scan[d]], cfg.sigma_v0 ** 2])


def predict(st, P, dx, cfg):
    """Advance dx metres. Works on [t, u] (cv) or [t, u, g] (ca), decided by
    the state length. Q is the integrated-noise covariance of the model
    linearised around the step's mean speed s (= u for cv)."""
    u = st[1]
    if u <= MIN_SPEED:
        return None
    if len(st) == 2:                                        # cv
        F = np.array([[1.0, -dx / u ** 2], [0.0, 1.0]])#state transition Jacobian. It describes how the state evolves from one time step to the next.

        #calculates process noise covariance.
        #Why is process noise needed?
        # Even if the model assumes constant speed, real vehicles may accelerate or decelerate slightly.
        # The parameter q_u controls how much speed variation is allowed.
        Q = cfg.q_u * np.array([[dx ** 3 / (3 * u ** 4), -dx ** 2 / (2 * u ** 2)],
                                [-dx ** 2 / (2 * u ** 2), dx]])
        return np.array([st[0] + dx / u, u]), F @ P @ F.T + Q, F
        #returns:
        # 1. Predicted state.
        # 2. Predicted covariance. --The covariance prediction is: P^- = FPF^T+Q
        # 3. State transition matrix.
        # This is the similar to the standard Kalman prediction equation.


    g = st[2]                                               # ca
    s = u + g * dx / 2.0              # mean speed over the step (midpoint rule)
    u_new = u + g * dx
    if s <= MIN_SPEED or u_new <= MIN_SPEED:
        return None
    s2, s4 = s * s, s ** 4
    F = np.array([[1.0, -dx / s2, -dx ** 2 / (2 * s2)],
                  [0.0, 1.0, dx],
                  [0.0, 0.0, 1.0]])
    Qu = cfg.q_u * np.array([[dx ** 3 / (3 * s4), -dx ** 2 / (2 * s2), 0.0],
                             [-dx ** 2 / (2 * s2), dx, 0.0],
                             [0.0, 0.0, 0.0]])
    Qg = cfg.q_g * np.array([[dx ** 5 / (20 * s4), -dx ** 4 / (8 * s2), -dx ** 3 / (6 * s2)],
                             [-dx ** 4 / (8 * s2), dx ** 3 / 3, dx ** 2 / 2],
                             [-dx ** 3 / (6 * s2), dx ** 2 / 2, dx]])
    return np.array([st[0] + dx / s, u_new, g]), F @ P @ F.T + Qu + Qg, F

    ##Qu represents uncertainty caused by speed variation.
    # Qg represents uncertainty caused by drift variation.


def update(st, P, z, S):
    '''This is the measurement update step of the Kalman filter.
    Prediction estimates where the vehicle should be. 
    Measurement update corrects that estimate using the actual sensor detection.'''
    #calculates the Kalman gain.
    K = P[:, 0] / S
    nu = z - st[0]
    return st + K * nu, P - np.outer(K, K) * S, nu
    #Mathematically:
        # X^+=X^-+K*nu
        # This corrects the estimated time, speed and, for CA, drift.
        # 1. X^+ is the updated state mean.
        # 2. X^- is the predicted state mean.
        # 3. K is the Kalman gain.
        # 4. \nu is the measurement residual.
        # 5. S is the measurement noise covariance.
        # Mathematically:
        # P^+= P^- - KSK^T
        # This reduces state uncertainty after incorporating the new measurement.
        # 1. P^+ is the updated state covariance.
        # 2. P^- is the predicted state covariance.
        # 3. K is the Kalman gain.
        # 4. S is the measurement noise covariance.
        # 5. K^T is the transpose of the Kalman gain.


def gate(D, j, t_pred, S, pool=None):
    '''checks if a detection is within the gate of the predicted detection time.
    If it is, it is considered for association.'''
    ids, zs = D.by_scan[j]
    h = GATE * math.sqrt(S)
    lo = np.searchsorted(zs, t_pred - h, "left")
    hi = np.searchsorted(zs, t_pred + h, "right")
    return [int(d) for d in ids[lo:hi] if pool is None or int(d) in pool]


def run_filter(D, dids, cfg, keep=False):
    """Score a hard set of detections from its first one to the bridge exit."""
    #sorts the detections by sensor_id and detection_time.
    dids = sorted(dids, key=lambda d: (D.scan[d], D.z[d]))

    #Reject invalid tracks: if the detections are not from the same sensor 
    if not dids or len({D.scan[d] for d in dids}) != len(dids):
        return -math.inf, None

   # Initialize the track: uses the first detection to initialize the track.
    first = dids[0]
    j0 = int(D.scan[first])
    st, P = init_state(D, first, cfg)
    llr = birth_llr(D, first, cfg)
    #The initial track score is its birth likelihood ratio.

    # Create detection lookup: a dictionary that maps each sensor_id to the detection_id.
    hit_at = {int(D.scan[d]): d for d in dids[1:]}
    hist = [(j0, st, P, None, None, None)]
    for j in range(j0 + 1, D.n_scans):#for every sensor after the first one
        pr = predict(st, P, D.scan_x[j] - D.scan_x[j - 1], cfg)#predict the state and covariance for the next sensor
        if pr is None:
            return -math.inf, None
        sp, Pp, F = pr
        d = hit_at.get(j)#get the detection_id for the next sensor

        if d is None:#if the there is no detection: The track receives a missed-detection penalty.
            st, P = sp, Pp
            llr += D.miss_ll[j]
        else:
            S = Pp[0, 0] + D.sig2[j]
            st, P, nu = update(sp, Pp, D.z[d], S)
            llr += hit_ll(D, j, nu, S)
            #The state is updated and the hit likelihood is added to the track score.
        if keep:
            hist.append((j, st, P, sp, Pp, F))
            #stores the predicted and updated states.
    return llr, (hist if keep else None)
    #returns the track score and the history of the track.(Overall track likelihood, Optional filtering history.)


def rts(hist):
    """Rauch-Tung-Striebel smoother; returns smoothed state means.
    Mathematically:
        C_k=P_kF_k^T(P_{k+1}^-)^{-1}
        x_k=x_k^+ + C_k(x_{k+1}^- - x_{k+1}^+)
        # 1. C_k is the covariance correction matrix.
        # 2. P_k is the predicted covariance.
        # 3. F_k is the state transition matrix.
        # 4. P_{k+1}^- is the predicted covariance of the next time step.
        # 5. x_k is the predicted state mean.
        # 6. x_k^+ is the updated state mean.
        # 7. x_{k+1}^- is the predicted state mean of the next time step.
        # 8. x_{k+1}^+ is the updated state mean of the next time step. 
        """
    xs = [h[1] for h in hist]
    for k in range(len(hist) - 2, -1, -1):
        _, sp, Pp, F = hist[k + 1][2:]
        C = hist[k][2] @ F.T @ np.linalg.pinv(Pp)

        xs[k] = hist[k][1] + C @ (xs[k + 1] - sp)
        #This corrects the earlier state estimate using information from the next smoothed state.
    return xs
    


# ==========================================================================
# MHT: hypothesis trees + global selection
# ==========================================================================
@dataclass
class Node:
    dids: tuple #Detection IDs assigned to the track
    st: np.ndarray #Current vehicle state
    P: np.ndarray #State covariance
    j: int #Current sensor index
    llr: float #Track likelihood ratio
#Each node is one branch in the hypothesis tree.

@dataclass
class Cand:
    dids: frozenset
    llr: float
    source: str #    Source of the candidate (MHT, JPDA or hybrid).
#Each candidate is one possible track hypothesis.
#A candidate represents a complete proposed vehicle track.
#Unlike a Node, a Cand represents a complete track candidate rather than an intermediate hypothesis.

def grow(D, root, cfg, pool=None):
    """Grow one tree to the bridge exit: at every sensor each node branches
    into MISS or HIT with any gated detection; keep the best `beam` nodes."""
    #1.Initialize the tree
    nodes = [root]
    #2.move through sensors
    for j in range(root.j + 1, D.n_scans):
        dx = D.scan_x[j] - D.scan_x[j - 1]
        nxt = []
        for n in nodes:
            pr = predict(n.st, n.P, dx, cfg)#At each sensor, the algorithm predicts the vehicle state.
            if pr is None:
                continue
            sp, Pp, _ = pr
            S = Pp[0, 0] + D.sig2[j]
            #3.Create MISS hypothesis
            nxt.append(Node(n.dids, sp, Pp, j, n.llr + D.miss_ll[j]))
            #4.Create HIT hypotheses
            for d in gate(D, j, sp[0], S, pool):#finds all detctions insode the gate
                s2, P2, nu = update(sp, Pp, D.z[d], S)#Updates the vehicle state.
                nxt.append(Node(n.dids + (d,), s2, P2, j, n.llr + hit_ll(D, j, nu, S)))
                #A new branch is created with that detection.
        if not nxt:
            return []

        #5.apply beam pruning
        nxt.sort(key=lambda n: -n.llr)#Sorts hypotheses from highest to lowest likelihood.
        nodes = [n for n in nxt if n.llr >= nxt[0].llr - cfg.llr_window][:cfg.beam]#Keeps only the best hypotheses.
    return nodes


def tomht_candidates(D, cfg, pool=None, source="mht"):
    """One tree per detection that could be a vehicle's first detection. Roots
    deep inside the bridge are skipped: their earlier misses already cost
    more than llr_window."""
    out = {}
    #1.consider evry detection as possible vehicle birth
    for d in range(len(D.z)):#for every detection_time
        if pool is not None and d not in pool:
            continue
        #2.Calculate birth likelihood
        llr0 = birth_llr(D, d, cfg)
        if llr0 < -cfg.llr_window:
            continue
        #3.Intiliaze a root node
        st, P = init_state(D, d, cfg)
        #4. Grow the hypothesis tree
        for leaf in grow(D, Node((d,), st, P, int(D.scan[d]), llr0), cfg, pool):
            #5. store valid candidate
            key = frozenset(leaf.dids)

            #A candidate is retained if:    Positive likelihood ratio,     Minimum number of unique sensor positions.
            if key not in out and leaf.llr > 0 and n_positions(D, key) >= cfg.min_positions:
                out[key] = Cand(key, leaf.llr, source)

    return list(out.values())#returns all unique candidate tracks.


def conflict_groups(cands):
    '''This function groups candidates that share detections.'''
    #every candidate belongs to its own group.
    parent = list(range(len(cands))) #used a Disjoint Set Union (DSU) data structure.

    def find(i): #nested fucntion
        '''Finds the root parent of a candidate.'''
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}#Stores which candidate first used each detection.

    #Checks all detections belonging to each candidate.
    #If two candidates share a detection, they are merged into the same conflict group.
    for i, c in enumerate(cands):
        for d in c.dids:
            if d in owner:
                parent[find(i)] = find(owner[d])
            else:
                owner[d] = i
    groups = {}
    for i in range(len(cands)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()) #returns the independent conflict groups


def beam_hyps(cands, members, cfg):
    '''This function generates possible combinations of candidates within one conflict group.'''

    hyps = [(0.0, frozenset(), frozenset())]        # empty = "all clutter"
    for i in sorted(members, key=lambda i: -cands[i].llr):
        grown = list(hyps)
        for sc, used, ids in hyps:
            if used.isdisjoint(cands[i].dids):#checks whether the candidate shares any detection with the currently selected tracks., If not, it can be added
                grown.append((sc + cands[i].llr, used | cands[i].dids, ids | {i}))#likelihood scores are accumulated:
        
        grown.sort(key=lambda h: -h[0])#sorts hypotheses and retains the best ones.
        hyps = [h for h in grown if h[0] >= grown[0][0] - cfg.llr_window][:MAX_HYPS]
    return {ids: sc for sc, _, ids in hyps}


def map_solve(cands, members):
    """max sum(LLR_i y_i)  s.t. each detection used at most once, y binary."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError:
        return None
    dets = sorted(set().union(*(cands[i].dids for i in members)))
    row = {d: r for r, d in enumerate(dets)}
    r_, c_ = [], []
    for c, i in enumerate(members):
        for d in cands[i].dids:
            r_.append(row[d])
            c_.append(c)
    A = csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(dets), len(members)))
    res = milp(c=-np.array([cands[i].llr for i in members]),
               constraints=LinearConstraint(A, 0, 1),
               integrality=np.ones(len(members)), bounds=Bounds(0, 1))
    if not res.success:
        return None
    return frozenset(i for c, i in enumerate(members) if res.x[c] > 0.5)


def select_global(cands, cfg):
    """Best consistent set of tracks (MAP) + approximate P(track is real)."""
    sel, p = [], np.zeros(len(cands))
    for members in conflict_groups(cands):
        hyps = beam_hyps(cands, members, cfg)
        best = map_solve(cands, members) if len(members) > 1 else frozenset(members)
        if best is None:
            best = max(hyps, key=hyps.get)
        hyps.setdefault(best, sum(cands[i].llr for i in best))
        keys = list(hyps)
        sc = np.array([hyps[k] for k in keys])
        w = np.exp(sc - sc.max())
        w /= w.sum()
        for k, wk in zip(keys, w):
            for i in k:
                p[i] += wk
        sel.extend(best)
    return sel, p


# ==========================================================================
# JPDA (one scan = one sensor)
# ==========================================================================
@dataclass
class JTrack:
    tid: int
    st: np.ndarray
    P: np.ndarray
    llr: float
    peak: float
    confirmed: bool = False
    alive: bool = True
    hist: list = field(default_factory=list)       # [(scan, st, P)]


def clusters(info):
    parent = {t: t for t in info}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for tid, (_, nu) in info.items():
        for d in nu:
            if d in owner:
                parent[find(tid)] = find(owner[d])
            else:
                owner[d] = tid
    groups = {}
    for tid in info:
        groups.setdefault(find(tid), []).append(tid)
    return list(groups.values())


def joint_assoc(cl, info, j, D):
    """Enumerate feasible joint events -> marginal beta per track, track-level
    likelihood ratio, and ambiguity = 1 - max P(joint event)."""
    opts, lr = [], {}
    for tid in cl:
        S, nu = info[tid]
        o = [(None, D.miss_ll[j])] + [(d, hit_ll(D, j, v, S)) for d, v in nu.items()]
        opts.append(o)
        m = max(w for _, w in o)
        lr[tid] = m + math.log(sum(math.exp(w - m) for _, w in o))
    events = []

    def rec(k, used, lw, assign):
        if len(events) >= MAX_EVENTS:
            raise OverflowError
        if k == len(cl):
            events.append((lw, tuple(assign)))
            return
        for d, w in opts[k]:
            if d is None or d not in used:
                rec(k + 1, used | {d} if d is not None else used, lw + w, assign + [d])

    try:
        rec(0, frozenset(), 0.0, [])
    except OverflowError:                 # fallback: independent PDA per track
        b = {}
        for tid, o in zip(cl, opts):
            ws = np.array([w for _, w in o])
            ws = np.exp(ws - ws.max())
            ws /= ws.sum()
            b[tid] = {d: p for (d, _), p in zip(o, ws)}
        return b, lr, 1.0
    lws = np.array([e[0] for e in events])
    w = np.exp(lws - lws.max())
    w /= w.sum()
    b = {tid: {} for tid in cl}
    for (_, a), p in zip(events, w):
        for tid, d in zip(cl, a):
            b[tid][d] = b[tid].get(d, 0.0) + p
    return b, lr, 1.0 - w.max()


def pda_update(tr, S, nu, b):
    K = tr.P[:, 0] / S
    b0 = b.get(None, 0.0)
    nb = sum(p * nu[d] for d, p in b.items() if d is not None)
    spread = sum(p * nu[d] ** 2 for d, p in b.items() if d is not None) - nb ** 2
    KK = np.outer(K, K)
    tr.st = tr.st + K * nb
    tr.P = b0 * tr.P + (1.0 - b0) * (tr.P - KK * S) + KK * spread


def manage(tracks, cfg):
    """Confirm when llr_window above clutter; delete when llr_window below the
    track's best; merge tracks whose states are within one std of each other."""
    live = [tr for tr in tracks if tr.alive]
    for tr in live:
        tr.peak = max(tr.peak, tr.llr)
        tr.confirmed = tr.confirmed or tr.llr >= cfg.llr_window
        if tr.llr < tr.peak - cfg.llr_window:
            tr.alive = False
    live = sorted((tr for tr in live if tr.alive), key=lambda tr: tr.st[0])
    for a, b in zip(live, live[1:]):
        if a.alive and b.alive:
            dlt = a.st - b.st
            if dlt @ np.linalg.solve(a.P + b.P, dlt) < 1.0:
                (a if a.llr < b.llr else b).alive = False


def jpda(D, cfg):
    tracks, beta, zones = [], {}, []
    for j in range(D.n_scans):
        dx = D.scan_x[j] - D.scan_x[j - 1] if j else 0.0
        info = {}
        for tr in tracks:
            if not tr.alive:
                continue
            pr = predict(tr.st, tr.P, dx, cfg)
            if pr is None:
                tr.alive = False
                continue
            tr.st, tr.P, _ = pr
            S = tr.P[0, 0] + D.sig2[j]
            info[tr.tid] = (S, {d: D.z[d] - tr.st[0] for d in gate(D, j, tr.st[0], S)})
        explained = {}
        for cl in clusters(info):
            b, lr, amb = joint_assoc(cl, info, j, D)
            for tid in cl:
                S, nu = info[tid]
                pda_update(tracks[tid], S, nu, b[tid])
                tracks[tid].llr += lr[tid]
                for d, p in b[tid].items():
                    if d is not None:
                        beta[(d, tid)] = p
                        explained[d] = explained.get(d, 0.0) + p
            if len(cl) > 1 and amb > cfg.amb_thresh:
                zones.append((j, set(cl)))
        manage(tracks, cfg)
        for d in D.by_scan[j][0]:              # births from unexplained detections
            d = int(d)
            if explained.get(d, 0.0) >= 0.5:
                continue
            llr0 = birth_llr(D, d, cfg)
            if llr0 < -cfg.llr_window:
                continue
            st, P = init_state(D, d, cfg)
            tracks.append(JTrack(len(tracks), st, P, llr0, llr0))
            beta[(d, len(tracks) - 1)] = 1.0 - explained.get(d, 0.0)
        for tr in tracks:
            if tr.alive:
                tr.hist.append((j, tr.st.copy(), tr.P.copy()))
    return tracks, beta, zones


def jpda_hard(D, tracks, beta, cfg):
    """Each detection to its most likely confirmed track (beta >= 0.5),
    at most one detection per track per sensor."""
    conf = {tr.tid for tr in tracks if tr.confirmed}
    best = {}
    for (d, tid), p in beta.items():
        if tid in conf and p >= 0.5 and p > best.get(d, (0.0, -1))[0]:
            best[d] = (p, tid)
    per = {}
    for d, (p, tid) in best.items():
        cur = per.setdefault(tid, {}).get(D.scan[d])
        if cur is None or p > cur[0]:
            per[tid][D.scan[d]] = (p, d)
    out = {}
    for tid, by in per.items():
        dids = frozenset(d for _, d in by.values())
        if n_positions(D, dids) >= cfg.min_positions:
            out[tid] = dids
    return out


# ==========================================================================
# Hybrid: JPDA everywhere, MHT inside ambiguous zones, one MILP at the end
# ==========================================================================
def merge_zones(zones):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, tids in zones:
        tl = list(tids)
        for t in tl[1:]:
            parent[find(t)] = find(tl[0])
    out = {}
    for j, tids in zones:
        r = find(next(iter(tids)))
        j0, s = out.get(r, (j, set()))
        out[r] = (min(j0, j), s | tids)
    return list(out.values())


def hybrid(D, cfg):
    tracks, beta, zones = jpda(D, cfg)
    hard = jpda_hard(D, tracks, beta, cfg)
    cands = [Cand(s, run_filter(D, s, cfg)[0], "jpda") for s in hard.values()]
    owner = {d: tid for tid, s in hard.items() for d in s}
    lock = 1.0 - cfg.amb_thresh
    n_zone = n_skip = 0
    for j0, tids in merge_zones(zones):
        if len(tids) > MAX_ZONE_TRACKS:
            n_skip += 1
            continue
        n_zone += 1
        pool = {int(d) for d in np.flatnonzero(D.scan >= j0)
                if not (d in owner and owner[d] not in tids
                        and beta[(d, owner[d])] >= lock)}
        for tid in tids:                       # roots = JPDA states before the zone
            snap = [h for h in tracks[tid].hist if h[0] < j0]
            if not snap:
                continue
            jj, st, P = snap[-1]
            prefix = frozenset(d for d in hard.get(tid, ()) if D.scan[d] <= jj)
            for leaf in grow(D, Node((), st, P, jj, 0.0), cfg, pool):
                full = prefix | frozenset(leaf.dids)
                if full and n_positions(D, full) >= cfg.min_positions:
                    cands.append(Cand(full, run_filter(D, full, cfg)[0], "zone-mht"))
        cands += tomht_candidates(D, cfg, pool, "zone-mht")   # vehicles new in the zone
    uniq = {}
    for c in cands:
        if c.llr > 0 and c.dids not in uniq:
            uniq[c.dids] = c
    cands = list(uniq.values())
    sel, p = select_global(cands, cfg)
    print(f"  hybrid: {n_zone} zones solved by MHT, {n_skip} too large (left to JPDA)")
    return cands, sel, p


# ==========================================================================
# Output
# ==========================================================================
def entry_time(t, u, g, back):
    """Time at the bridge entry, `back` metres before a state (t, u, g)."""
    s = u - g * back / 2.0            # mean speed over the back-extrapolated span
    return t - back / s if s > MIN_SPEED else np.nan


def write_output(D, cands, sel, p, cfg, outdir):
    x0, x1 = D.scan_x[0], D.scan_x[-1]
    has_g = cfg.motion == "ca"
    traj, assoc, veh = [], [], []
    vid = 0
    for i in sorted(sel, key=lambda i: min(D.z[d] for d in cands[i].dids)):
        c = cands[i]
        llr, hist = run_filter(D, c.dids, cfg, keep=True)
        if hist is None:
            continue
        xs = rts(hist)
        hit_at = {int(D.scan[d]) for d in c.dids}
        for h, s in zip(hist, xs):
            traj.append(dict(vehicle_id=vid, sensor_id=D.scan_sensor[h[0]],
                             position=D.scan_x[h[0]], detected=h[0] in hit_at,
                             t_smoothed=s[0], speed_smoothed=s[1],
                             drift_smoothed=s[2] if has_g else np.nan))
        for d in c.dids:
            assoc.append(dict(vehicle_id=vid, event_id=D.det.at[d, "event_id"],
                              sensor_id=D.scan_sensor[D.scan[d]],
                              detection_time=D.det.at[d, "detection_time"]))
        g0 = xs[0][2] if has_g else 0.0
        t_in = entry_time(xs[0][0], xs[0][1], g0, D.scan_x[hist[0][0]] - x0)
        t_out = xs[-1][0]
        veh.append(dict(vehicle_id=vid, entry_time=t_in, exit_time=t_out,
                        mean_speed=(x1 - x0) / (t_out - t_in) if t_out > t_in else np.nan,
                        entry_speed=xs[0][1], exit_speed=xs[-1][1],
                        mean_drift=float(np.mean([s[2] for s in xs])) if has_g else np.nan,
                        n_hits=len(c.dids), llr=llr, p_track=p[i], source=c.source))
        vid += 1
    pd.DataFrame(traj).to_csv(os.path.join(outdir, "trajectories.csv"), index=False)
    pd.DataFrame(assoc, columns=["vehicle_id", "event_id", "sensor_id",
                               "detection_time"]).to_csv(os.path.join(outdir, "associations.csv"), index=False)
    pd.DataFrame(veh).to_csv(os.path.join(outdir, "vehicles.csv"), index=False)
    return vid, len(assoc)


# ==========================================================================
# Evaluation against the ground truth
# ==========================================================================
def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def print_metrics(title, res):
    print(f"  {title}")
    for k, v in res.items():
        print(f"    {k:34s} {v:.4g}" if isinstance(v, float) else f"    {k:34s} {v}")


def load_truth(cfg):
    truth = pd.read_csv(cfg.truth)
    dets_of = {v: [str(e) for e in json.loads(s)]
               for v, s in zip(truth.vehicle_id, truth.caused_detection_ids)}
    times_of = {v: {str(k): float(t) for k, t in json.loads(s).items()}
                for v, s in zip(truth.vehicle_id, truth.true_crossing_time_per_station)}
    owners = {}                       # event_id -> true vehicles (merged events: several)
    for v, ds in dets_of.items():
        for e in ds:
            owners.setdefault(e, set()).add(v)
    return truth, dets_of, times_of, owners


def load_detections(cfg):
    return (pd.read_csv(cfg.detections, dtype={"event_id": str, "sensor_id": str})
            .drop_duplicates("event_id"))


def detector_metrics(cfg):
    """Quality of the extracted detections against the ground truth,
    overall and per sensor. Independent of the tracker."""
    _, dets_of, times_of, owners = load_truth(cfg)
    det = load_detections(cfg)
    sensor_of = dict(zip(det.event_id, det.sensor_id))
    time_of = dict(zip(det.event_id, det.detection_time))
    t_start, t_end = det.detection_time.min(), det.detection_time.max()
    duration = t_end - t_start

    passages, detected = {}, {}       # true crossings / crossings with a detection
    for v, tt in times_of.items():
        for s, t in tt.items():
            if t_start <= t <= t_end:
                passages[s] = passages.get(s, 0) + 1
    for v, s in {(v, sensor_of[e]) for v, ds in dets_of.items() for e in ds if e in sensor_of}:
        detected[s] = detected.get(s, 0) + 1
    errors = {}                       # detection time - true crossing time
    for e, own in owners.items():
        if len(own) == 1 and e in sensor_of:          # merged detections skipped
            v, s = next(iter(own)), sensor_of[e]
            if s in times_of[v]:
                errors.setdefault(s, []).append(time_of[e] - times_of[v][s])

    rows = []
    for s, g in det.groupby("sensor_id"):
        own = [owners.get(e, set()) for e in g.event_id]
        n_veh = sum(1 for o in own if o)
        err = np.array(errors.get(s, []))
        rows.append(dict(
            sensor_id=s, longitudinal_position=g.longitudinal_position.iloc[0],
            true_passages=passages.get(s, 0), detected_passages=detected.get(s, 0),
            detections=len(g), vehicle_detections=n_veh, clutter=len(g) - n_veh,
            merged=sum(1 for o in own if len(o) > 1),
            bias=err.mean() if len(err) else np.nan,
            sigma_t=err.std(ddof=1) if len(err) > 1 else np.nan))
    t = pd.DataFrame(rows).sort_values("longitudinal_position").reset_index(drop=True)
    t["p_d"] = t.detected_passages / t.true_passages.where(t.true_passages > 0)
    t["precision"] = t.vehicle_detections / t.detections
    t["lam_fa"] = t.clutter / duration
    t.to_csv(os.path.join(cfg.outdir, "detector_metrics.csv"), index=False)

    all_err = np.concatenate([np.array(e) for e in errors.values()]) if errors else np.array([])
    rec = t.detected_passages.sum() / max(1, t.true_passages.sum())
    prec = t.vehicle_detections.sum() / max(1, t.detections.sum())
    print_metrics("detections vs ground truth", {
        "true passages": int(t.true_passages.sum()),
        "detected passages": int(t.detected_passages.sum()),
        "detection recall (P_D)": float(rec),
        "detection precision": float(prec),
        "detection F1": f1(rec, prec),
        "clutter detections": int(t.clutter.sum()),
        "false alarms per sensor per s": float(t.clutter.sum() / (len(t) * duration)),
        "merged detections": int(t.merged.sum()),
        "timing bias [s]": float(all_err.mean()) if len(all_err) else np.nan,
        "timing std [s]": float(all_err.std(ddof=1)) if len(all_err) > 1 else np.nan,
        "lowest sensor P_D": f"{t.p_d.min():.3g} (sensor {t.sensor_id[t.p_d.idxmin()]})",
    })
    print(f"    per-sensor table -> {os.path.join(cfg.outdir, 'detector_metrics.csv')}")


def evaluate(cfg, outdir):
    """Tracker output against the ground truth: track level and sensor level."""
    _, dets_of, _, owners = load_truth(cfg)
    n_true = {v: len(ds) for v, ds in dets_of.items()}
    true_ids = {v for v, n in n_true.items() if n >= MIN_HITS}
    det = load_detections(cfg)
    asc = pd.read_csv(os.path.join(outdir, "associations.csv"),
                      dtype={"event_id": str, "sensor_id": str})
    size = asc.groupby("vehicle_id").size()

    ov = {}                           # (track, true vehicle) -> shared detections
    for v, e in zip(asc.vehicle_id, asc.event_id):
        for tv in owners.get(e, ()):
            ov[(v, tv)] = ov.get((v, tv), 0) + 1
    match, used = {}, set()
    for (v, tv), n in sorted(ov.items(), key=lambda kv: -kv[1]):
        if v in match or tv in used:
            continue
        if n / size[v] >= MIN_PURITY and n / n_true[tv] >= MIN_COMPLETE:
            match[v] = tv
            used.add(tv)
    n_tracks_per_true = {}
    for (v, tv), n in ov.items():
        if n >= 2:
            n_tracks_per_true[tv] = n_tracks_per_true.get(tv, 0) + 1
    asc["correct"] = [match.get(v) in owners.get(e, ())
                      for v, e in zip(asc.vehicle_id, asc.event_id)]
    correct = int(asc.correct.sum())
    clutter = set(det.event_id) - set(owners)

    # track level
    tp = sum(1 for tv in match.values() if tv in true_ids)
    fp = len(size) - len(match)
    fn = len(true_ids - used)
    t_prec, t_rec = tp / max(1, tp + fp), tp / max(1, tp + fn)

    # detection level (association)
    d_prec = correct / max(1, len(asc))
    d_rec = correct / max(1, len(owners))
    clutter_rejected = len(clutter - set(asc.event_id))
    d_acc = (correct + clutter_rejected) / max(1, len(det))

    # sensor level (association)
    det["vehicle"] = det.event_id.isin(owners.keys())
    per = pd.DataFrame({
        "longitudinal_position": det.groupby("sensor_id").longitudinal_position.first(),
        "vehicle_detections": det[det.vehicle].groupby("sensor_id").size(),
        "assigned": asc.groupby("sensor_id").size(),
        "correct": asc.groupby("sensor_id").correct.sum(),
    }).fillna(0).sort_values("longitudinal_position")
    per["precision"] = per.correct / per.assigned.where(per.assigned > 0)
    per["recall"] = per.correct / per.vehicle_detections.where(per.vehicle_detections > 0)
    per["f1"] = 2 * per.precision * per.recall / (per.precision + per.recall)
    per.to_csv(os.path.join(outdir, "sensor_metrics.csv"), index_label="sensor_id")

    print_metrics("tracks vs ground truth", {
        "true vehicles": len(true_ids),
        "output tracks": len(size),
        "track TP / FP / FN": f"{tp} / {fp} / {fn}",
        "track precision": t_prec,
        "track recall": t_rec,
        "track F1": f1(t_prec, t_rec),
        "fragmented vehicles": sum(1 for tv in true_ids if n_tracks_per_true.get(tv, 0) > 1),
        "detection precision": d_prec,
        "detection recall": d_rec,
        "detection F1": f1(d_prec, d_rec),
        "detection accuracy": d_acc,
        "lowest sensor recall": f"{per.recall.min():.3g} (sensor {per.recall.idxmin()})",
    })
    print(f"    per-sensor table -> {os.path.join(outdir, 'sensor_metrics.csv')}")


# ==========================================================================
def run_method(D, cfg, method):
    if method == "mht":
        cands = tomht_candidates(D, cfg)
        sel, p = select_global(cands, cfg)
    elif method == "jpda":
        tracks, beta, _ = jpda(D, cfg)
        cands = [Cand(s, run_filter(D, s, cfg)[0], "jpda")
                 for s in jpda_hard(D, tracks, beta, cfg).values()]
        sel, p = list(range(len(cands))), np.full(len(cands), np.nan)
    elif method == "hybrid":
        cands, sel, p = hybrid(D, cfg)
    else:
        sys.exit(f"unknown method: {method}")
    return cands, sel, p


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python bridge_mtt.py params.yaml")
    cfg = load_config(sys.argv[1])
    #print(cfg)
    D = load(cfg)
    print(f"{len(D.z)} detections, {D.n_scans} sensors | motion model: {cfg.motion}")
    if cfg.truth:
        os.makedirs(cfg.outdir, exist_ok=True)
        #detector_metrics(cfg)
    for method in cfg.methods:
        outdir = os.path.join(cfg.outdir, method)
        os.makedirs(outdir, exist_ok=True)
        cands, sel, p = run_method(D, cfg, method)
        n, k = write_output(D, cands, sel, p, cfg, outdir)
        print(f"[{method}] {n} vehicles, {k} associated detections -> {outdir}")
        if cfg.truth:
            evaluate(cfg, outdir)


if __name__ == "__main__":
    main()