''' 
Bridge Multi-Vehicle Tracker - a probabilistic multi-vehicle tracker for the bridge

This tracker is a probabilistic multi-vehicle tracker for the bridge.

run: python prob_mulitvehicle_tracker.py [--detections <detections.csv>] [--ground-truth <ground_truth.csv>] [--output <output.csv>]

all parameters are in the CONFIG block. lets start with basic version.'''

import argparse
from dataclasses import dataclass


import pandas as pd
import numpy as np
import math
from src.research.scripts.vehicle_speed_est import Config
from src.track.tracker import CFG


##============================================================
## CONFIG
##============================================================
class CONFIG:
    #---model constants (from bridge and measuring specifications)
    sigma_t = 0.005    ## sec - timestamp noise std
    p_d = 0.95         ## no -- per sensor detection prob
    t_merge = 0.10     ## sec -- merging window time 
    lambda_FALSE = 0.1 ## noise rate per sensor per min
    V_MIN = 5.0        ## m/s statistical gate speed for min 18kmph(*3.6)
    V_MAX = 50.0       ## m/s statistical gate speed for max 180kmph(*3.6)
    mu_v  = 16.7       ## m/s mean of speed prior
    sigma_v = 2.0      ## stf for speed prior

    #---- provisional (have to tune emprically)
    epsilon_t = 0.500  ## sec - physical estimated gate time - 50 ms
    n_neighbours = 4   ## forward neighbour positions used to calculate average speed
    delta_v = 5.0      ## m/s -- avg spped tolerance
    k_gate = 4.0       ## statistical gate width multiplier 
    k_prune = 100.0    ## MHT pruning constant
    N_scan  = 3        ## n_scan commit horizn --positions


# ==================================================================   
# Step-1 --- Loading
# ==================================================================   

def load_detections(paths):
    '''load detetctions csv'''
    frames = [pd.read_csv(p) for p in paths]
    det = pd.concat(frames, ignore_index=True)
    #det = det.sort_values('detction_time').reset_index(drop=True) 
    #optional beacuse the behaviour of the tarck can go forward and backward 
    det = det.drop_duplicates(subset=['event_id']).reset_index(drop=True)
    det['did'] = np.arrange(len(det)) #internal contiguous id w.r.t frames
    return det

def build_geomtery(det):
    '''making sensors and positions setup in ordered list '''
    positions = np.sort(det['longitudinal_position'].unique())
    pos_index = {p: i for i, p in enumerate(positions) }

    #co-location: sensors sharing a same lateral position
    colocated = {}
    for p, grp in det.groupby('longitudinal_position'):
        colocated[p] = sorted(grp['sensor_id'].unique().tolist())
    return positions, pos_index, colocated   

# =================================================================== 
# Step-2 ---- physcially possible limts of velocity gating
# =================================================================== 

def possible_gate_limit(t_i, dx, pos_times_i):
    ''' Return only possible vehicles tracks w.r.t to physically possible limits'''
    #dx/V_MAX - eps <= (tj-ti) <= dx/V_MIN + eps
    #  t_j - dx/V_MIN - eps <= t_i <= t_j - dx/V_MAX + eps
    low = t_i - dx / CONFIG.V_MIN - CONFIG.epsilon_t
    high = t_i - dx / CONFIG.V_MAX + CONFIG.epsilon_t
    t = pos_times_i['t']
    a = np.searchsorted(t, low, side = 'left')
    b = np.searchsorted(t, high, side = 'right')
    return pos_times_i['did'][a:b], t[a:b]

def velocities_tolerance(cand, delta_v):
    ''' velocity should falll in the tolerance limit'''
    cand = sorted(cand, key = lambda c: c[0])
    velocities = []
    current = [cand[0]]
    for c in cand[1:]:
        if c[0] - current[-1][0] <= delta_v:
            current.append(c)
        else:
            velocities.append(current)
            current = [c]
    velocities.append(current)
    # only keeping velocities with >=1 member 
    return velocities 

def generate_track_seeds(det, positions, pos_index, pos_times):
    '''avergaing the velocties over N forward neighbout sensors '''

    track_seeds = []

    for i, x_j in enumerate(positions):
        # downstream detections at this postion
        down = pos_times[x_j]
        for t_j, did_j in zip(down['t'], down['did']):
            # gather candidate velocities to the N forwrad group of sensors
            # 'forward' = smaller x already passed; we look positions after x_i
            cand = [] #(v, did_i, x_i)
            for i in range(max(0, i - CONFIG.n_neighbours), i):
                x_i = positions[i]
                dx = x_j - x_i
                if dx <= 0:
                    continue
                dids_i, ts_i = possible_gate_limit(t_i, dx, pos_times[x_i])
                for did_i, t_i in zip(dids_i, ts_i):
                    dt = t_j - t_i
                    if dt <= 0:
                        continue
                    v = dx / dt
                    if CONFIG.V_MIN <= v <= CONFIG.V_MAX:
                        cand.append((v, int(did_i), x_j))
            if not cand:
                continue
            
            # cluster candidate velocities; keep tight limits
            velocities = velocities_tolerance(cand, CONFIG.delta_v)
            for v in velocities:
                vs = [c[0] for c in v]
                u_0 = float(np.mean(vs))
                #here we reject implausible speeds based on statstical prior
                #keeps with the tolerance mu +/- 4*sigma
                if abs(u_0 - CONFIG.mu_v) > 4 * CONFIG.sigma_v
                    continue

                t_0 = t_j - (x_j) / u_0  #lets say entry time referenced to x=0
                track_seeds.append(
                        {'det_ids': [int(did_j)] + [c[1] for c in v],
                        'u_0': u_0,
                        'x_0': 0.0,
                        't_0': t_0,
                        'anchor_det_id': int(did_j),
                        'anchor_x': float(x_j),
                        'anchor_t': float(t_j),})
    return track_seeds

# ==================================================
# step 4 - Line fit and mht tree generation 
# ==================================================

@dataclass
class Node:
    det_ids: list                             # dtections in this track-do-far
    a: float                                  # fitted t at x = 0
    u: float                                  # fitted speed
    logscore: float
    positions_used: set                       # set bccoz no duplicates
    fork_depth: int = 0                       # positions since last fork(for N-scan)

def predict_time(a, u, x):
    return a + x / u

def measuremnt_loglikliehood(t_abs, t_hat):
    z = (t_abs - t_hat) / CONFIG.sigma_t
    return -0.5 * z * z - math.log(CONFIG.sigma_t * math.sqrt(2 * math.pi))

def speed_prior_loglikliehood(u):
    z = (u - CONFIG.mu_v) / CONFIG.sigma_v
    return -0.5 * z * z - math.log(CONFIG.sigma_v * math.sqrt(2 * math.pi))

def fit_line(points):
    ''' least squares fit, solves and try to fit the èpoints in a line  '''
    xs = np.array([p[0] for p in points], float)
    ts = np.array([p[1] for p in points], float)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return None
    A = np.vstack([np.ones_like/(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    a,b = coef  #t = a +bx ; b =1/u
    if b<= 0:
        return None
    u = 1.0/b
    return a,u   # a = t at x=0 (entry time ref), u = speed

def extend_track(track_seed, det, positions, pos_index, pos_times):
    ''' '''

    detection_x = det.set_index('did')['logitudinal_postion'].to_dict()
    detection_t = det.set_index('did')['detection_time'].to_dict()

    points = [(detection_x[d], detection_t[d]) for d in track_seed['dids']]
    fit = fit_line(points)
    a, u = fit if fit else (track_seed['t0'], track_seed['u0'])

    dids = list(track_seed['dids'])
    used_pos = set(detection_x[d] for d in dids)
    logscore = speed_prior_loglikliehood(u)

    #loop for postions on the bridge
    for x_s in positions:
        if x_s in used_pos:
            continue
        t_hat = predict_time(a, u, x_s)
        pt = pos_times[x_s]
        low_end = t_hat - CONFIG.k_gate * CONFIG.sigma_t
        high_end = t_hat + CONFIG.k_gate * CONFIG.sigma_t
        ai = np.searchsorted(pt['t'], low_end, 'left')
        bi = np.searchsorted(pt['t'], high_end, 'right')
        survived = list(zip(pt['did'][ai:bi], pt['t'][ai:bi]))

        if not survived:
            logscore += math.log(1-CONFIG.p_d)
            continue
        # picks the best fitting survivor,

        best_detection_id, best_t = min(survived, key = lambda s: abs(s[1] - t_hat))
        used_pos.add(x_s)
        dids.append(int(best_detection_id))

        for did_s, t_s in survived:
            if int(did_s) != int(best_detection_id) and abs(t_s - best_t) <= CONFIG.epsilon_t:
                dids.append(int(did_s))
        
        fit = fit_line([(detection_x[d], detection_x[d]) for d in dids ])

        if fit:
            a,u = fit

        logscore += (measuremnt_loglikliehood(best_t, predict_time(a, u, x_s)) 
                        + math.log(CONFIG.p_d))

    logscore += speed_prior_loglikliehood(u)
    return [Node(dids, a, u, logscore, used_pos)]






def parse_args():
    p = argparse.ArgumentParser(description='multi vehicle tracker- a probablistic multi vehcile tracker')
    p.add_argument('--detction', help='path to the detctions csv (required)', required=True)
    p.add_argument('--ground_truth', help= 'path to the ground trutvh csv file (optional)')
    p.add_argument('--outdir', help='path to output directory')
    return p.parse_args()




def main():
    args = parse_args()
    

if __name__ == "__main__":
    main()