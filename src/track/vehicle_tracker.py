from enum import Flag
import os
import sys
import math
import argparse
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd



class cfg:
    #model constants with some intital assumptions
    SIGMA_T = 0.05 # timestamp noise std
    P_D = 0.95 #per sensor detection prob
    T_MERGE = 0.10 # MERGE WINDOW (s)
    LAMBDA_FALSE = 0.2 #Clutter rate per sensor per minute
    V_MIN = 11.0 #m/s -->39.6km/hr physical gate min
    V_MAX = 50.0 #m/s -->180km/hr physical gate max
    MU_V = 22.5 # SPEED prior mean -->81km/hr
    SIGMA_V = 2.0 # SPEED prior std --->7.2 km/hr

    # testing and tunable -- empirically
    EPSILON_T = 0.800 # physical gate time slack (widened from 50 ms)
    N_NEIGHBOURS = 4 # forward neighbour postions used for tracks(should
                    # avoid the co-located sensor on the pther side )
    DELTA_V = 2.0 # vehciel sep
    K_GATE = 4.0 # statistical gate width multiplier (k * sigma_t)
    K_PRUNE = 100.0 # MHT realtive pruning constant
    N_SCAN = 3 # number of scan commit horizons (positions)


##################################################################
# step 1- Loader
################################################################## 
def load_detections(paths):
    """concatenate windows given in args, deduplicate, build per postion index """
    frames = [pd.read_csv(p) for p in paths]
    det = pd.concat(frames, ignore_index = True)
    #det = det.sort_values("detection_time").reset_index(drop=True)
    #deduplicate exact seam duplicates (same event_id appearing twice)
    det = det.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    det["did"] = np.arange(len(det))
    return det

def build_geometry(det):
    ''' matching postions and sensors, '''
    positions = np.sort(det['longitudinal_position'].unique())
    pos_index = {p: i for i, p in enumerate (positions) }
    #co-located: sensors sharing same postion but opposite sides of the bridge
    colocated ={}
    for p, grp in det.groupby('longitudinal_position'):
        colocated[p] = sorted(grp['sensor_id'].unique().tolist())
    return positions, pos_index, colocated

def per_position_times(det, positions):
    '''for each postion:'''
    idx = {}
    for p in positions:
        sub = det[det['longitudinal_position'] == p].sort_values('detection_time')
        idx[p] = {
            't': sub['detection_time'].to_numpy(),
            'did': sub['did'].to_numpy(),
        }
    return idx


##################################################################
# step 2- Physical gate + track building
################################################################# 
def physical_gate_parameters(t_i, dx, pos_times_i):
    '''return detection_ids at postion i that satisfy the physical gate 
    for a later sensor detection at time t_i with dx = x_j -x_i >0. '''
   # dx/vmax -eps <= t_j - t_i <= dx/min + eps
   # => t_j -dx/vmin -eps <= t_i <= t_j - dx/vmax + eps
    low = t_i -dx/cfg.V_MIN - cfg.EPSILON_T
    high = t_i -dx/cfg.V_MAX + cfg.EPSILON_T 

    t = pos_times_i['t']
    a = np.searchsorted(t, low, side ='left')
    b= np.searchsorted(t, high, side='right')

    return pos_times_i['did'][a:b], t[a:b]

def  cluster_velocities(cand, delta_v):
    """Greedy 1-D clustering by velocity; returns list of clusters (each a list
    of (v,did,x)). A cluster is 'tight' if members are within delta_v."""

    #keep the speed guesses that agree with each other, 
    # because agreement is evidence they're all really describing the same car 
    # and throw out or separate the odd one out.
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

def generate_seeds(det, positions, pos_index, pos_times):
    ''' fileters the avergae tracks over N forward neighbours'''
    seeds = []

    #Calculating the velocities based on the known distances between the sensors
    for i, x_i in enumerate(positions):
        #downstream detections at this postion in the loop
        down  = pos_times[x_i]
        for t_i, did_i in zip(down['t'], down['did']):
            # gather candidate from the WIM gate
            cand =[] #(v, did_j, x_j)
            for j in range(max(0, i - cfg.N_NEIGHBOURS), i):
                x_j = positions[j]
                dx= x_j -x_i
                if dx>=0:
                    continue
                dids_j, ts_j = physical_gate_parameters(t_i, dx, pos_times[x_j])
                for did_j, t_j in zip(dids_j, ts_j):
                    dt = t_j - t_i
                    if dt>=0:
                        continue
                    v = dx/dt
                    if cfg.V_MIN <= v <= cfg.V_MAX:
                        cand.append((v, int(did_j), x_j))
            if not cand:
                continue
            #cluster candidate velocities, keep tight clusters
            clusters = cluster_velocities(cand, cfg.DELTA_V)
            for cl in clusters:
                v = [c[0] for c in cl]
                u0 = float(np.mean(v))
                #reject speeds which are outside limits of traffic prior
                # also removes false pairings like one car 
                # to another by averaging over the n-neighbours
                if abs(u0 - cfg.MU_V) > 4 * cfg.SIGMA_V:
                    continue
                #what time would the car have been at position 0 gives exactly this formula
                t0 = t_i - (x_i) / u0 #entry time referenced to x=0
            

                seeds.append({
                    'dids': [int(did_i)] + [c[1] for c in cl], #list of detection_ids +all the downstream detections in the cluster
                    'u0':u0,
                    'x0':0.0,
                    't0':t0,
                    'anchor_did': int(did_i),
                    'anchor_x': float(x_i),
                    'anchor_t': float(t_i)
                })
    return seeds

#############################################################
#step 4 & 5 ---> track line fit and MHT tree
#############################################################

def fit_line(points):
    '''least squares fit of t = t0 + x/u given (x,t) points. Returns (a (t0--->at x0 =0))
     solves t = a + b*x , u= 1/b'''
    xs = np.array([p[0] for p in points], float)
    ts = np.array ([p[1] for p in points], float)

    if len(xs) < 2 or np.ptp(xs) == 0: #if peak to peak == 0
        return None
    A = np.vstack([np.ones_like(xs), xs]).T

    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    a,b = coef #t = a+bx (b=1/u)
    if b <= 0:
        return None
    u = 1.0 / b
    return a,  u  #a=t (at x= 0 entry time reference), u = speed

@dataclass
class Node:
    dids: list # detections in this track so far
    a: float
    u : float 
    logscore: float
    positions_used: set
    fork_depth: int= 0

def predict_t(a, u, x):
    return a + x/ u

def measurement_loglik(t_abs, t_hat):
    z = (t_abs - t_hat) / cfg.SIGMA_T
    return -0.5 * z * z -math.log(cfg.SIGMA_T * math.sqrt(2 * math.pi))

def speed_prior_loglikliehood(u):
    z = (u - cfg.MU_V) / cfg.SIGMA_V
    return -0.5 * z * z - math.log(cfg.SIGMA_V * math.sqrt(2 * math.pi))


def extend_track(seed, det, positions, pos_index, pos_times):
    ''' grow  one seed into one track by best fit attachment with refit '''
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()

    pts = [(did_x[d], did_t[d]) for d in seed["dids"]]
    fit = fit_line(pts)
    a, u = fit if fit else (seed["t0"], seed["u0"])

    dids = list(seed["dids"])
    used_pos = set(did_x[d] for d in dids)
    logscore = speed_prior_loglikliehood(u)

    for x_s in positions:
        if x_s in used_pos:
            continue
        t_hat = predict_t(a, u, x_s)
        pt = pos_times[x_s]
        low = t_hat - cfg.K_GATE * cfg.SIGMA_T
        high = t_hat + cfg.K_GATE * cfg.SIGMA_T
        ai = np.searchsorted(pt['t'], low, 'left')
        bi = np.searchsorted(pt['t'], high, 'right')
        surv = list(zip(pt['did'][ai:bi], pt['t'][ai:bi]))

        if not surv:
            logscore += math.log(1 - cfg.P_D) #miss
            continue
        #pick the best fitting survivor, then combine its co-located sensor
        best_did, best_t = min(surv, key =lambda s: abs(s[1] - t_hat))
        used_pos.add(x_s)
        dids.append(int(best_did))
        for did_s, t_s in surv:
            if int(did_s) != int(best_did) and abs(t_s - best_t) <= cfg.EPSILON_T:
                dids.append(int(did_s))
        fit = fit_line([(did_x[d], did_t[d]) for d in dids])
        if fit:
            a, u = fit
        logscore += (measurement_loglik(best_t, predict_t(a, u, x_s))
                        + math.log(cfg.P_D))

    logscore += speed_prior_loglikliehood(u)
    return [Node(dids, a, u, logscore, used_pos)]

#########################################################################
# step -6 JPDA (per clusters contested)
########################################################################

def build_tracks(seeds, det, positions, pos_index, pos_times):
    tracks =[]
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
    '''remove near duplicate tracks'''
    tracks.sort(key =lambda n: (len(n.dids), n.logscore), reverse=True)
    kept = []
    for t in tracks:
        ds = set(t.dids)
        dup = False
        for k in kept:
            ks =set(k.dids)
            inter = len(ds & ks)
            #overlap relative to the smaller trackkkkkkkkkkkkkkkk
            if inter / max(1, min(len(ds), len(ks))) > 0.3:
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept




def contested_clusters(tracks):
    '''Gorup tracks that share >=1 detction (connectivity)'''              
    parent_track = list(range(len(tracks)))
    def find(i):
        while parent_track[i] != i:
            parent_track[i] = parent_track[parent_track[i]]; i = parent_track[parent_track[i]]
        return i
    def union(i,j):
        parent_track[find(i)] = find(j)
    sets = [set(t.dids) for t in tracks]

    for i in range(len(tracks)):
        for j in range (i + 1, len(tracks)):
            if sets[i] & sets[j]:
                union(i,j)
    groups ={}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(i)                 
    return list(groups.values())



def merge_allowed(ti, tj, d , tracks,did_x):
    x = did_x[d]
    thi = predict_t(tracks[ti].a, tracks[ti].u, x)
    thj = predict_t(tracks[tj].a, tracks[tj].u, x)
    return abs(thi-thj) <= cfg.T_MERGE


def enumerate_cluster(cl_dids, claim, weight, tracks, did_x, detections):
    '''enumerate exclusive assignements (+ allowed merges) then marginalise '''
    dids = sorted(cl_dids)
    #options per detection 
    options = {}
    for d in dids:
        opts = [('t', ti) for ti in claim[d]]
        opts.append(('clutter', None))
        #allowed merge options; claiming track pairs within t_merge
        for ti, tj in itertools.combinations(claim[d], 2):
            if merge_allowed(ti, tj, d, tracks, did_x):
                opts.append(('merge', (ti, tj)))
        options[d] = opts
    
    #enumerate joint events 
    keys = dids
    total = 1
    for d in keys:
        total *= len(options[d])
    marg = {d: {} for d in keys}
    Z = 0.0

    if total >200000:
        #fallback: per detection independent normalisation
        for d in keys:
            num = {}
            for kind, val in options[d]:
                if kind == 't':
                    p = weight[(d, val)]
                    num[val] = num.get(val, 0) + p
                elif kind == 'clutter':
                    num['clutter'] = num.get('clutter', 0) + cfg.LAMBDA_FALSE
                else:
                    for ti in val:
                        num[ti] = num.get(ti, 0) + weight[(d, ti)]
            s = sum(num.values()) or 1.0
            marg[d] = {k: v / s for k, v in num.items()}
        return marg
    
    for combo in itertools.product(*[options[d] for d in keys]):
        #exclusivity check 
        prob = 1.0
        assign = {}
        for d, (kind, val) in zip(keys, combo):
            if kind == 't':
                prob *= weight[(d, val)]
                assign[d] = [val]
            elif kind == 'clutter':
                prob *= cfg.LAMBDA_FALSE
                assign[d] = ['clutter']
            else:
                prob *= weight[(d, val[0])] * weight[(d, val[1])]
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


def jpda(tracks, detections, postions):
    did_x = detections.set_index('did')['longitudinal_position'].to_dict()
    associations = {}
    clusters = contested_clusters(tracks)

    for cl in clusters:
        #detctions only involved in this cluster
        cl_dids = set()
        for track in cl:
            cl_dids |= set(tracks[track].dids) #union set operation(unites tracks)
        
        #per (did, track) base likelihood weight
        weight ={} # (did, track) -> weight
        for track in cl:
            t = tracks[track]
            for detection in t.dids:
                x = did_x[detection]
                t_hat = predict_t(t.a, t.u, x)
                #using measuremnt likliehood as weight
                # time of this detection
                t_obs = detections.set_index('did')['detection_time'].to_dict()[detection]
                weight[(detection, track)] = math.exp(measurement_loglik(t_obs, t_hat))

        #expand joint events over the clusters detections
        #each detction should be -> one of its claiming tracks or clutter
        claim  = {d: [track for track in cl if (d, track) in weight] for d in cl_dids}
        associations.update(enumerate_cluster(cl_dids, claim, weight, tracks, did_x, detections))

    return associations, clusters


    
def parse_args():
    p = argparse.ArgumentParser(
        description="Bridge vehicle tracker —> timestamp-only multi-target "
                    "tracking (gating -> seeding -> extension -> JPDA).")
    p.add_argument("--detections",
               required =True, help="path to the detections CSV (required)")
    p.add_argument("--gt", "--ground-truth", dest="gt", default=None,
                   help="optional ground-truth CSV (used only for scoring)")
    p.add_argument("--outdir", default=".",
                   help="directory for any output files (default: current)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.detections):
        sys.exit(f'detections file not found: {args.detections}')
    if args.gt and not os.path.isfile(args.gt):
        sys.exit(f'ground-truth file not found: {args.gt}')
    os.makedirs(args.outdir, exist_ok=True)

    detections = load_detections([args.detections])
    positions, pos_index, colocated = build_geometry(detections)
    pos_times = per_position_times(detections, positions)

    print(f'loaded {len(detections)} detctions, {len(positions)} positions,'
            f"{detections['sensor_id'].nunique()} sensors")

    seeds = generate_seeds(detections, positions, pos_index, pos_times)
    print(f'generated {len(seeds)} raw tracks')

    #MHT IS applied here
    tracks = build_tracks(seeds, detections, positions, pos_index, pos_times)
    print(f'built {len(tracks)} tracks after deduplication')


    soft,clusters = jpda(tracks, detections, positions)
    print(f'JPDA: {len(clusters)} contested clusters,'
            f'{len(soft)} detections with soft assignments')

if __name__ == '__main__':
    main()
