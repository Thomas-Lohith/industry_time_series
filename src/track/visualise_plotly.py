"""
Visualise the PROCESS of tracking (not just the final result).

Axes are swapped per request: x = distance (position), y = time.
Because sensors are co-located in pairs separated by gaps, a tracked vehicle
climbs a STAIRCASE: a short vertical tread at each junction (two co-located
detections, near-identical time) joined by diagonal risers (between-junction
travel). The animation replays one track's extension step by step:
predict -> gate -> attach -> re-fit, one junction at a time.

Run:  python visualize_process.py <detections.csv> [ground_truth.csv] [track_rank]
      track_rank (optional): which track to animate, 0 = longest (default 0)
Output: tracking_process.html
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import tracker as T


def record_extension(seed, det, positions, pos_times):
    """Re-run the greedy extension but RECORD each step for animation.
    Returns list of step dicts."""
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()

    pts = [(did_x[d], did_t[d]) for d in seed["dids"]]
    fit = T.fit_line(pts)
    a, u = fit if fit else (seed["t0"], seed["u0"])
    dids = list(seed["dids"])
    used_pos = set(did_x[d] for d in dids)

    steps = []
    # initial state
    steps.append(dict(kind="seed", a=a, u=u, dids=list(dids),
                      x_s=None, t_hat=None, gate=None, attached=[]))

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

        # PREDICT + GATE step (before attaching)
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
        # ATTACH + REFIT step
        steps.append(dict(kind="attach", a=a, u=u, dids=list(dids),
                          x_s=x_s, t_hat=t_hat, gate=(lo, hi),
                          attached=attached))
    return steps


def _step_traces(st, det, did_x, did_t, did_e, X, Tt, xspan):
    """Build the FIXED set of traces for one step. Every step returns the SAME
    NUMBER of traces in the SAME ORDER (empty where not applicable) so Plotly
    animation can map frame i's trace k onto the initial trace k. Mismatched
    trace counts between frames is a common cause of blank/broken animations."""
    txs = [did_x[d] for d in st["dids"]]
    tys = [did_t[d] for d in st["dids"]]
    tline = st["a"] + xspan / st["u"]

    if st["x_s"] is not None:
        pred_x, pred_y = [st["x_s"]], [st["t_hat"]]
        lo, hi = st["gate"]
        gate_x, gate_y = [st["x_s"], st["x_s"]], [lo, hi]
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


def build_animation(det, steps, out):
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()
    did_e = det.set_index("did")["event_id"].to_dict()

    X = det["longitudinal_position"].to_numpy()
    Tt = det["detection_time"].to_numpy()
    xspan = np.array([X.min(), X.max()])
    xpad = 0.03 * (X.max() - X.min())
    tpad = 0.03 * (Tt.max() - Tt.min())

    frames = []
    for si, st in enumerate(steps):
        traces = _step_traces(st, det, did_x, did_t, did_e, X, Tt, xspan)
        title = (f"step {si}/{len(steps)-1}: {st['kind']}"
                 + (f"  @ x={st['x_s']}m" if st["x_s"] is not None else ""))
        frames.append(go.Frame(data=traces, name=str(si),
                               layout=go.Layout(title_text=title)))

    # initial data = the first frame's traces (same count/order as every frame)
    fig = go.Figure(data=_step_traces(steps[0], det, did_x, did_t, did_e, X, Tt, xspan),
                    frames=frames)
    fig.update_xaxes(title_text="distance / longitudinal position (m)",
                     range=[X.min() - xpad, X.max() + xpad])
    fig.update_yaxes(title_text="time (s)",
                     range=[Tt.min() - tpad, Tt.max() + tpad])
    fig.update_layout(
        title_text="Tracking process — vehicle climbs the staircase (predict/gate/attach/refit)",
        height=800, width=1100, hovermode="closest",
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=1.15,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=700, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
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
    # do NOT use scattergl; write full plotly.js inline for offline reliability
    fig.write_html(out, include_plotlyjs=True, auto_play=False)


def main():
    det_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
    rank = 0
    for a in sys.argv[2:]:
        if a.isdigit():
            rank = int(a)

    det = T.load_detections([det_path])
    positions, pos_index, colocated = T.build_geometry(det)
    pos_times = T.per_position_times(det, positions)
    seeds = T.generate_seeds(det, positions, pos_index, pos_times)
    tracks = T.build_tracks(seeds, det, positions, pos_index, pos_times)

    # pick the track to animate: rank-th longest
    tracks_sorted = sorted(tracks, key=lambda t: len(t.dids), reverse=True)
    target = tracks_sorted[min(rank, len(tracks_sorted) - 1)]

    # find the seed that produced it (best overlap)
    best_seed, best_ov = None, -1
    tset = set(target.dids)
    for s in seeds:
        ov = len(set(s["dids"]) & tset)
        if ov > best_ov:
            best_ov, best_seed = ov, s

    steps = record_extension(best_seed, det, positions, pos_times)
    out = "tracking_process.html"
    build_animation(det, steps, out)
    print(f"animated track rank {rank} (u={target.u:.1f}, "
          f"{len(target.dids)} dets, {len(steps)} steps) -> {out}")


if __name__ == "__main__":
    main()