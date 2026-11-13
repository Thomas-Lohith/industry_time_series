"""
Stage-by-stage debug runner for vehicle_tracker_staged.py

Takes a small time slice of the detections CSV and prints what every stage
returns, so each function can be checked by eye.

  python debug_stages.py --detections detections_window_0.csv --t-max 30
  python debug_stages.py --detections detections_window_0.csv --t-min 60 --t-max 90
  python debug_stages.py --detections detections_window_0.csv --rows 40     # first 40 rows only
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import tracker_staged as m

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def ev(D, d):
    """Readable label for a detection id: event_id."""
    return D.df.at[d, "event_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=30.0)
    ap.add_argument("--rows", type=int, default=None, help="use only the first N rows of the slice")
    ap.add_argument("--show", type=int, default=6, help="rows to print per table")
    ap.add_argument("--subset-out", default="debug_subset.csv")
    args = ap.parse_args()
    cfg = m.Cfg()
    N = args.show

    # ---- 0. slice the CSV ------------------------------------------------
    full = pd.read_csv(args.detections)
    sub = full[(full.detection_time >= args.t_min) & (full.detection_time < args.t_max)]
    if args.rows:
        sub = sub.head(args.rows)
    sub.to_csv(args.subset_out, index=False)
    D = m.load(args.subset_out)
    banner(f"0. LOAD   slice t in [{args.t_min}, {args.t_max}) -> {len(D.t)} detections "
           f"(saved to {args.subset_out})")
    print(f"positions ({len(D.positions)}): {np.round(D.positions, 2).tolist()}")
    print(f"sensors per position: {[len(s) for s in D.sensors_at]}")
    print(D.df[["event_id", "sensor_id", "longitudinal_position", "detection_time"]].head(N)
          .rename_axis("did").to_string())

    # ---- 1. physical gating ---------------------------------------------
    pairs = m.physical_gating(D, cfg)
    banner(f"1. physical_gating   -> {len(pairs)} event pairs (speed in "
           f"[{cfg.v_min}, {cfg.v_max}] m/s)")
    print("Each row: two detections at different positions with a plausible speed.")
    rows = [dict(ev1=ev(D, a), ev2=ev(D, b), x1=D.x[a], x2=D.x[b],
                 t1=D.t[a], t2=D.t[b], speed=v) for a, b, v in pairs[:N]]
    print(pd.DataFrame(rows).round(3).to_string(index=False) if rows else "(none)")

    # ---- 2. velocity scoring -------------------------------------------
    scored = m.velocity_scoring(pairs, cfg)
    banner(f"2. velocity_scoring  -> {len(scored)} pairs kept, {len(pairs) - len(scored)} dropped "
           f"(|v - {cfg.mu_v}| > {cfg.v_sigmas}*{cfg.sigma_v})")
    print("Sorted best first (highest speed log-likelihood).")
    rows = [dict(ev1=ev(D, a), ev2=ev(D, b), speed=v, loglik=ll) for a, b, v, ll in scored[:N]]
    print(pd.DataFrame(rows).round(3).to_string(index=False) if rows else "(none)")

    # ---- 3a. trace ONE tree, position by position ------------------------
    banner("3a. mht_tree - trace of the tree grown from the best-scored seed pair")
    if scored:
        d1, d2, v, _ = scored[0]
        print(f"seed: {ev(D, d1)} -> {ev(D, d2)}  (speed {v:.2f} m/s)")
        fit = m.fit_line(D, (d1, d2))
        nodes = [m.Node((d1, d2), fit[0], fit[1], 0.0)]
        seen = {D.pos[d1], D.pos[d2]}
        centre = (D.x[d1] + D.x[d2]) / 2
        order = sorted((k for k in range(len(D.positions)) if k not in seen),
                       key=lambda k: abs(D.positions[k] - centre))
        for step, k in enumerate(order[:N + 4], 1):
            nxt = [c for n in nodes for c in m.children(D, n, k, cfg)]
            nxt.sort(key=lambda n: -n.logscore)
            best = nxt[0].logscore
            kept = [n for n in nxt if n.logscore >= best - cfg.k_prune][:cfg.branch]
            print(f"  step {step:>2} x={D.positions[k]:7.2f}: {len(nxt)} children -> "
                  f"kept {len(kept)} | " + " ; ".join(
                      f"{len(n.dids)} dets, score {n.logscore:7.2f}, u={n.u:5.2f}" for n in kept))
            nodes = kept
        print("  (only the first steps are shown; the real tree continues to all positions)")

    # ---- 3b. all candidate tracks ---------------------------------------
    tracks = m.mht_tree(D, scored, cfg)
    banner(f"3b. mht_tree  -> {len(tracks)} candidate tracks (leaves of all trees, after gap fill)")
    rows = []
    for i, t in enumerate(tracks):
        rows.append(dict(track=i, n_dets=len(t.dids),
                         n_positions=len({D.pos[d] for d in t.dids}),
                         entry_time_a=t.a, speed_u=t.u, score=t.score))
    tdf = pd.DataFrame(rows)
    print(tdf.round(3).head(max(N, 12)).to_string(index=False) if rows else "(none)")

    if not tracks:
        print("\nNo tracks - nothing further to show.")
        return

    # ---- 4. JPDA -------------------------------------------------------
    P, p_clutter, p_track, stats = m.jpda(D, tracks, cfg)
    banner(f"4. jpda  -> {len(stats)} conflict groups; P matrix shape {P.shape} "
           f"(detections x tracks)")
    groups = m.conflict_groups(tracks)
    for gi, members in enumerate(groups[:N]):
        hyps = m.group_hypotheses(tracks, members, cfg)
        sc = np.array([h[0] for h in hyps])
        w = np.exp(sc - sc.max()); w /= w.sum()
        print(f"  group {gi}: tracks {members} | {len(hyps)} global hypothesis(es)")
        for (s, ids), p in list(zip(hyps, w))[:4]:
            print(f"      P={p:.3f}  score={s:9.2f}  tracks={sorted(ids)}")
    print(f"\n  P(track) for the first tracks: "
          f"{ {i: round(float(p_track[i]), 3) for i in range(min(N, len(tracks)))} }")
    print(f"  detections: {int((p_clutter < 0.05).sum())} clearly assigned, "
          f"{int((p_clutter > 0.95).sum())} clearly clutter, "
          f"{int(((p_clutter >= 0.05) & (p_clutter <= 0.95)).sum())} uncertain")
    row_sums = P.sum(axis=1)
    print(f"  sanity: max row-sum of P = {row_sums.max():.3f} (must be <= 1)")
    d_show = int(np.flatnonzero(P.sum(axis=1) > 0)[0])
    nz = {int(i): round(float(P[d_show, i]), 3) for i in np.flatnonzero(P[d_show])}
    print(f"  example row: detection {ev(D, d_show)} -> {nz}, clutter={p_clutter[d_show]:.3f}")

    # ---- 5. output ------------------------------------------------------
    outdir = "debug_out"
    os.makedirs(outdir, exist_ok=True)
    traj, n = m.write_output(D, tracks, P, p_track, cfg, outdir)
    banner(f"5. write_output  -> {n} vehicles, {len(traj)} trajectory rows (files in {outdir}/)")
    print(pd.read_csv(os.path.join(outdir, "vehicles.csv")).round(3).head(N).to_string(index=False))
    print("\nfirst trajectory (ordered by position):")
    if len(traj):
        first = traj[traj.vehicle_id == traj.vehicle_id.iloc[0]]
        print(first.head(N + 2).round(3).to_string(index=False))


if __name__ == "__main__":
    main()