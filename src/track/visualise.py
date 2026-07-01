"""
Visualise the tracker on a window.
Produces position-vs-time plots (constant speed => straight line).

Run:  python visualize.py <detections.csv> [ground_truth.csv]
Outputs PNGs next to this script.
"""

import sys
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tracker as T


def plot_raw(det, out):
    """All detections in position-time space -- the raw data."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(det["detection_time"], det["longitudinal_position"],
               s=8, c="#444", alpha=0.6)
    ax.set_xlabel("detection time (s)")
    ax.set_ylabel("longitudinal position (m)")
    ax.set_title(f"Raw detections ({len(det)} points) — each vehicle is a line")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_truth(det, gt, out):
    """Ground-truth trajectories: each vehicle's true crossing times per station."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(det["detection_time"], det["longitudinal_position"],
               s=8, c="#ccc", alpha=0.5, zorder=1)
    # sensor_id -> position
    sid_pos = det.drop_duplicates("sensor_id").set_index("sensor_id")["longitudinal_position"].to_dict()
    cmap = plt.get_cmap("tab20")
    for i, (_, row) in enumerate(gt.iterrows()):
        cross = row["true_crossing_time_per_station"].replace("'", '"')
        import json
        d = json.loads(cross)
        xs, ys = [], []
        for sid, t in d.items():
            sid = int(sid)
            if sid in sid_pos:
                xs.append(t); ys.append(sid_pos[sid])
        order = np.argsort(ys)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        ax.plot(xs, ys, "-", color=cmap(i % 20), lw=1.5, zorder=2)
        ax.text(xs[0], ys[0], f"v{int(row['vehicle_id'])}", fontsize=7,
                color=cmap(i % 20))
    ax.set_xlabel("detection time (s)")
    ax.set_ylabel("longitudinal position (m)")
    ax.set_title(f"Ground truth — {len(gt)} vehicles")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_tracks(det, tracks, out):
    """Estimated tracks: detections coloured by track, fitted line drawn."""
    fig, ax = plt.subplots(figsize=(14, 7))
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()
    ax.scatter(det["detection_time"], det["longitudinal_position"],
               s=8, c="#ddd", alpha=0.5, zorder=1)
    cmap = plt.get_cmap("tab20")
    xspan = np.array([det["longitudinal_position"].min(),
                      det["longitudinal_position"].max()])
    for i, t in enumerate(tracks):
        col = cmap(i % 20)
        txs = [did_t[d] for d in t.dids]
        tys = [did_x[d] for d in t.dids]
        ax.scatter(txs, tys, s=18, color=col, zorder=3)
        # fitted line t = a + x/u  ->  plot as (t, x)
        tline = t.a + xspan / t.u
        ax.plot(tline, xspan, "-", color=col, lw=1, alpha=0.7, zorder=2)
    ax.set_xlabel("detection time (s)")
    ax.set_ylabel("longitudinal position (m)")
    ax.set_title(f"Estimated tracks — {len(tracks)} tracks "
                 f"(coloured points = track membership, lines = fitted speed)")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_merge_zoom(det, tracks, gt, out):
    """Zoom on the merge region (vehicles 8 & 9 in window 0, ~t=305)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    did_x = det.set_index("did")["longitudinal_position"].to_dict()
    did_t = det.set_index("did")["detection_time"].to_dict()
    # window around merge
    m = det[(det["detection_time"] > 303) & (det["detection_time"] < 320)]
    ax.scatter(m["detection_time"], m["longitudinal_position"],
               s=30, c="#bbb", alpha=0.7, zorder=1)
    cmap = plt.get_cmap("tab10")
    ci = 0
    for t in tracks:
        txs = [did_t[d] for d in t.dids if 303 < did_t[d] < 320]
        tys = [did_x[d] for d in t.dids if 303 < did_t[d] < 320]
        if len(txs) >= 3:
            ax.scatter(txs, tys, s=45, color=cmap(ci % 10), zorder=3,
                       label=f"track u={t.u:.1f}")
            ci += 1
    ax.set_xlabel("detection time (s)")
    ax.set_ylabel("longitudinal position (m)")
    ax.set_title("Merge zoom (vehicles 8 & 9): lines touch low, fan apart")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    det_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else None

    det = T.load_detections([det_path])
    positions, pos_index, colocated = T.build_geometry(det)
    pos_times = T.per_position_times(det, positions)
    seeds = T.generate_seeds(det, positions, pos_index, pos_times)
    tracks = T.build_tracks(seeds, det, positions, pos_index, pos_times)
    print(f"{len(det)} detections, {len(tracks)} tracks")

    plot_raw(det, "01_raw_detections.png")
    plot_tracks(det, tracks, "03_estimated_tracks.png")
    if gt_path:
        gt = pd.read_csv(gt_path)
        plot_truth(det, gt, "02_ground_truth.png")
        plot_merge_zoom(det, tracks, gt, "04_merge_zoom.png")
    print("saved plots: 01_raw / 02_ground_truth / 03_estimated_tracks / 04_merge_zoom")


if __name__ == "__main__":
    main()