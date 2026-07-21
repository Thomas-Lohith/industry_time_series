"""
Interactive Plotly visualisation of the tracker.
ONE run produces TWO separate HTML files:

  1. <overview>.html  -- estimated tracks vs ground truth (hover/zoom/toggle)
  2. <process>.html   -- step-by-step animation of one track forming
                         (predict -> gate -> attach -> re-fit)

Axes: x = distance (longitudinal position), y = time.

Run:
  python visualize_plotly.py DETECTIONS.csv [--gt GT.csv] [--rank N]
         [--outdir DIR] [--overview-out NAME.html] [--process-out NAME.html]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tracker as T


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------
def run_tracker(det_path):
    det = T.load_detections([det_path])
    positions, pos_index, colocated = T.build_geometry(det)
    pos_times = T.per_position_times(det, positions)
    seeds = T.generate_seeds(det, positions, pos_index, pos_times)
    tracks = T.build_tracks(seeds, det, positions, pos_index, pos_times)
    return det, positions, pos_index, pos_times, seeds, tracks


PALETTE = [f"hsl({int(h)},70%,45%)" for h in np.linspace(0, 330, 24)]


# ------------------------------------------------------------------
# VIEW 1 -- interactive overview (tracks vs ground truth)
# ------------------------------------------------------------------
def build_overview(det, tracks, gt, out):
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()
    did_e = det.set_index("did")["event_id"].to_dict()
    did_s = det.set_index("did")["sensor_id"].to_dict()

    has_gt = gt is not None
    ncols = 2 if has_gt else 1
    titles = ["Estimated tracks"] + (["Ground truth"] if has_gt else [])
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        shared_yaxes=True, horizontal_spacing=0.06)

    # background raw detections (distance x, time y)
    fig.add_trace(go.Scattergl(
        x=det["longitudinal_position"], y=det["detection_time"],
        mode="markers", marker=dict(size=4, color="lightgray"),
        name="all detections", hoverinfo="skip", showlegend=True), row=1, col=1)

    xspan = np.array([det["longitudinal_position"].min(),
                      det["longitudinal_position"].max()])
    for i, t in enumerate(tracks):
        col = PALETTE[i % len(PALETTE)]
        tpos = [did_x[d] for d in t.dids]
        ttim = [did_t[d] for d in t.dids]
        hover = [f"track {i}<br>u={t.u:.2f} m/s<br>event={did_e[d]}"
                 f"<br>sensor={did_s[d]}<br>t={did_t[d]:.3f}s<br>x={did_x[d]}m"
                 for d in t.dids]
        fig.add_trace(go.Scattergl(
            x=tpos, y=ttim, mode="markers",
            marker=dict(size=7, color=col),
            name=f"trk{i} u={t.u:.1f}", legendgroup=f"trk{i}",
            text=hover, hoverinfo="text"), row=1, col=1)
        tline = t.a + xspan / t.u
        fig.add_trace(go.Scattergl(
            x=xspan, y=tline, mode="lines",
            line=dict(color=col, width=1),
            legendgroup=f"trk{i}", showlegend=False,
            hoverinfo="skip"), row=1, col=1)

    if has_gt:
        sid_pos = det.drop_duplicates("sensor_id").set_index("sensor_id")[
            "longitudinal_position"].to_dict()
        fig.add_trace(go.Scattergl(
            x=det["longitudinal_position"], y=det["detection_time"],
            mode="markers", marker=dict(size=4, color="lightgray"),
            hoverinfo="skip", showlegend=False), row=1, col=2)
        for i, (_, row) in enumerate(gt.iterrows()):
            col = PALETTE[i % len(PALETTE)]
            d = json.loads(row["true_crossing_time_per_station"].replace("'", '"'))
            pos, tim = [], []
            for sid, tt in d.items():
                sid = int(sid)
                if sid in sid_pos:
                    pos.append(sid_pos[sid]); tim.append(tt)
            order = np.argsort(pos)
            pos = np.array(pos)[order]; tim = np.array(tim)[order]
            fig.add_trace(go.Scattergl(
                x=pos, y=tim, mode="lines+markers",
                line=dict(color=col, width=1.5), marker=dict(size=4),
                name=f"v{int(row['vehicle_id'])} u={row['speed']:.1f}",
                hovertext=[f"vehicle {int(row['vehicle_id'])}<br>"
                           f"u={row['speed']:.2f}" for _ in pos],
                hoverinfo="text"), row=1, col=2)

    fig.update_xaxes(title_text="distance / position (m)", row=1, col=1)
    if has_gt:
        fig.update_xaxes(title_text="distance / position (m)", row=1, col=2)
    fig.update_yaxes(title_text="time (s)", row=1, col=1)
    fig.update_layout(
        title=f"Tracker overview — {len(tracks)} estimated tracks"
              + (f" vs {len(gt)} true vehicles" if has_gt else ""),
        height=800, width=1500 if has_gt else 900,
        hovermode="closest", legend=dict(font=dict(size=9)))
    fig.write_html(out, include_plotlyjs="cdn")


# ------------------------------------------------------------------
# VIEW 2 -- process animation (one track forming)
# ------------------------------------------------------------------
def record_extension(seed, det, positions, pos_times):
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()
    pts = [(did_x[d], did_t[d]) for d in seed["dids"]]
    fit = T.fit_line(pts)
    a, u = fit if fit else (seed["t0"], seed["u0"])
    dids = list(seed["dids"])
    used_pos = set(did_x[d] for d in dids)
    steps = [dict(kind="seed", a=a, u=u, dids=list(dids),
                  x_s=None, t_hat=None, gate=None, attached=[])]
    for x_s in positions:
        if x_s in used_pos:
            continue
        t_hat = T.predict_t(a, u, x_s)
        pt = pos_times[x_s]
        lo = t_hat - T.CFG.K_GATE * T.CFG.SIGMA_T
        hi = t_hat + T.CFG.K_GATE * T.CFG.SIGMA_T
        ai = np.searchsorted(pt["t"], lo, "left")
        bi = np.searchsorted(pt["t"], hi, "right")
        surv = list(zip(pt["did"][ai:bi], pt["t"][ai:bi]))
        steps.append(dict(kind="predict", a=a, u=u, dids=list(dids),
                          x_s=x_s, t_hat=t_hat, gate=(lo, hi), attached=[]))
        if not surv:
            steps.append(dict(kind="miss", a=a, u=u, dids=list(dids),
                              x_s=x_s, t_hat=t_hat, gate=(lo, hi), attached=[]))
            continue
        best_did, best_t = min(surv, key=lambda s: abs(s[1] - t_hat))
        used_pos.add(x_s)
        attached = [int(best_did)]
        dids.append(int(best_did))
        for did_s, t_s in surv:
            if int(did_s) != int(best_did) and abs(t_s - best_t) <= T.CFG.EPSILON_T:
                dids.append(int(did_s)); attached.append(int(did_s))
        fit = T.fit_line([(did_x[d], did_t[d]) for d in dids])
        if fit:
            a, u = fit
        steps.append(dict(kind="attach", a=a, u=u, dids=list(dids),
                          x_s=x_s, t_hat=t_hat, gate=(lo, hi),
                          attached=attached))
    return steps


def _step_traces(st, did_x, did_t, did_e, X, Tt, xspan):
    txs = [did_x[d] for d in st["dids"]]
    tys = [did_t[d] for d in st["dids"]]
    tline = st["a"] + xspan / st["u"]
    if st["x_s"] is not None:
        pred_x, pred_y = [st["x_s"]], [st["t_hat"]]
        lo, hi = st["gate"]; gate_x, gate_y = [st["x_s"], st["x_s"]], [lo, hi]
    else:
        pred_x = pred_y = gate_x = gate_y = []
    if st["attached"]:
        ax = [did_x[d] for d in st["attached"]]
        ay = [did_t[d] for d in st["attached"]]
    else:
        ax = ay = []
    return [
        go.Scatter(x=X, y=Tt, mode="markers",
                   marker=dict(size=4, color="lightgray"),
                   hoverinfo="skip", name="all detections"),
        go.Scatter(x=txs, y=tys, mode="markers+lines",
                   marker=dict(size=8, color="crimson"),
                   line=dict(color="crimson", width=1),
                   name="track so far",
                   text=[did_e[d] for d in st["dids"]], hoverinfo="text"),
        go.Scatter(x=list(xspan), y=list(tline), mode="lines",
                   line=dict(color="crimson", width=1, dash="dot"),
                   name="fitted speed line", hoverinfo="skip"),
        go.Scatter(x=pred_x, y=pred_y, mode="markers",
                   marker=dict(size=13, color="royalblue", symbol="x"),
                   name="prediction t_hat", hoverinfo="skip"),
        go.Scatter(x=gate_x, y=gate_y, mode="lines",
                   line=dict(color="royalblue", width=8),
                   opacity=0.35, name="gate window", hoverinfo="skip"),
        go.Scatter(x=ax, y=ay, mode="markers",
                   marker=dict(size=15, color="gold",
                               line=dict(color="black", width=1)),
                   name="just attached", hoverinfo="skip"),
    ]


def build_process(det, steps, out):
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()
    did_e = det.set_index("did")["event_id"].to_dict()
    X = det["longitudinal_position"].to_numpy()
    Tt = det["detection_time"].to_numpy()
    xspan = np.array([X.min(), X.max()])
    xpad = 0.03 * (X.max() - X.min()); tpad = 0.03 * (Tt.max() - Tt.min())

    frames = []
    for si, st in enumerate(steps):
        traces = _step_traces(st, did_x, did_t, did_e, X, Tt, xspan)
        title = (f"step {si}/{len(steps)-1}: {st['kind']}"
                 + (f"  @ x={st['x_s']}m" if st["x_s"] is not None else ""))
        frames.append(go.Frame(data=traces, name=str(si),
                               layout=go.Layout(title_text=title)))

    fig = go.Figure(data=_step_traces(steps[0], did_x, did_t, did_e, X, Tt, xspan),
                    frames=frames)
    fig.update_xaxes(title_text="distance / position (m)",
                     range=[X.min() - xpad, X.max() + xpad])
    fig.update_yaxes(title_text="time (s)",
                     range=[Tt.min() - tpad, Tt.max() + tpad])
    fig.update_layout(
        title_text="Tracking process — predict/gate/attach/refit",
        height=800, width=1100, hovermode="closest",
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=1.15,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=700, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")])])],
        sliders=[dict(
            active=0, y=0, x=0.1, len=0.85,
            currentvalue=dict(prefix="step "),
            steps=[dict(method="animate", label=str(i),
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                             mode="immediate")])
                   for i in range(len(frames))])])
    fig.write_html(out, include_plotlyjs=True, auto_play=False)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Produce BOTH the interactive overview and the process "
                    "animation from one run.")
    p.add_argument("--detections", help="path to detections CSV")
    p.add_argument("--gt", default=None, help="optional ground-truth CSV")
    p.add_argument("--rank", type=int, default=0,
                   help="which track to animate in the process view (0=longest)")
    p.add_argument("--outdir", default=".", help="output directory")
    p.add_argument("--overview-out", default="tracker_interactive.html",
                   help="filename for the overview html")
    p.add_argument("--process-out", default="tracking_process.html",
                   help="filename for the process-animation html")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.detections):
        sys.exit(f"detections file not found: {args.detections}")
    if args.gt and not os.path.isfile(args.gt):
        sys.exit(f"ground-truth file not found: {args.gt}")
    os.makedirs(args.outdir, exist_ok=True)
    overview_path = os.path.join(args.outdir, args.overview_out)
    process_path = os.path.join(args.outdir, args.process_out)

    det, positions, pos_index, pos_times, seeds, tracks = run_tracker(args.detections)
    gt = pd.read_csv(args.gt) if args.gt else None

    # VIEW 1: overview
    build_overview(det, tracks, gt, overview_path)
    print(f"[overview] {len(det)} detections, {len(tracks)} tracks -> {overview_path}")

    # VIEW 2: process animation
    if not tracks:
        print("no tracks formed; skipping process animation")
        return
    tracks_sorted = sorted(tracks, key=lambda t: len(t.dids), reverse=True)
    if args.rank >= len(tracks_sorted):
        print(f"warning: rank {args.rank} >= {len(tracks_sorted)} tracks; using last")
    target = tracks_sorted[min(args.rank, len(tracks_sorted) - 1)]
    tset = set(target.dids)
    best_seed = max(seeds, key=lambda s: len(set(s["dids"]) & tset))
    steps = record_extension(best_seed, det, positions, pos_times)
    build_process(det, steps, process_path)
    print(f"[process ] track rank {args.rank} (u={target.u:.1f}, "
          f"{len(target.dids)} dets, {len(steps)} steps) -> {process_path}")


if __name__ == "__main__":
    main()