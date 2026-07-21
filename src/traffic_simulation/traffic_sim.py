#!/usr/bin/env python3
"""
Bridge Traffic Simulation (v1) — synthetic sensor detection generator.

Produces, per 15-minute window:
  - a DETECTION csv  (noisy per-sensor timestamp events: what the tracker sees)
  - a GROUND-TRUTH csv (which vehicle caused which detection: for scoring MHT/JPDA)

Design decisions (see spec v0.3):
  * Sensor geometry is read from a sensor CSV (source of truth) — NOT hardcoded.
  * Only the vertical channel / timestamp is used. Amplitude is NOT modelled.
  * v1 motion model = constant speed per vehicle (sampled from a distribution).
  * Travel direction is unused; a vehicle traverses the bridge extent and trips
    every sensor whose DIST_M lies within the span it actually travels.
  * No lateral sides in the real data -> each sensor is its own station;
    station miss probability = (1 - p_d).  Lateral mode is OFF in v1.

EVERY probability / spread in the config is an UNTUNED placeholder to calibrate.
This script does not validate them; it just uses them.
"""

import argparse
import csv
import json
import math
import os
import random

from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter

# ---------------------------------------------------------------------------
# Minimal YAML loader: try PyYAML, else fall back to a tiny parser for our
# specific, flat-ish config so the script runs with no dependencies.
# ---------------------------------------------------------------------------
def load_config(path):
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        raise SystemExit(
            "PyYAML not available or config unreadable. Install with "
            "`pip install pyyaml` (or pass --selftest to run without a config)."
        )


# ---------------------------------------------------------------------------
# Sensor loading
# ---------------------------------------------------------------------------
def load_sensors(cfg):
    s = cfg["sensors"]
    cols = s["columns"]
    sensors = []
    with open(s["csv_path"]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensors.append({
                "sensor_id": row[cols["id"]],
                "sensor_vertical": row[cols["vertical"]] ,
                "dist_m": float(row[cols["distance"]]),
                "span": row.get(cols["span"], ""),
            })
    # Order stations by position so detection sequences are positional.
    sensors.sort(key=lambda d: d["dist_m"])
    return sensors


# ---------------------------------------------------------------------------
# Vehicle generation
# ---------------------------------------------------------------------------
def truncated_normal(rng, mean, std, lo, hi):
    # simple rejection; bounds are wide so this terminates quickly
    for _ in range(1000):
        v = rng.gauss(mean, std)
        if lo <= v <= hi:
            return v
    return min(max(rng.gauss(mean, std), lo), hi)


def generate_vehicles(cfg, rng, window_seconds):
    """Poisson arrivals over the window; each vehicle gets a constant speed."""
    tr = cfg["traffic"]
    rate_per_s = tr["arrival"]["rate_veh_per_min"] / 60.0
    sp = tr["speed"]

    vehicles = []
    t = 0.0
    v_id = 0
    while True:
        # exponential inter-arrival gap (Poisson process)
        gap = rng.expovariate(rate_per_s) if rate_per_s > 0 else float("inf")
        t += gap
        if t >= window_seconds:
            break
        speed = truncated_normal(rng, sp["mean_mps"], sp["std_mps"],
                                 sp["min_mps"], sp["max_mps"])
        vehicles.append({"vehicle_id": v_id, "entry_time": t, "speed": speed})
        v_id += 1
    return vehicles


# ---------------------------------------------------------------------------
# Core: true crossing times, then apply detection model
# ---------------------------------------------------------------------------
def simulate_window(cfg, sensors, rng, window_id, window_seconds):
    det = cfg["detection"]
    bridge = cfg["bridge"]
    entry_pos = bridge.get("entry_position_m")
    exit_pos = bridge.get("exit_position_m")
    if entry_pos is None:
        entry_pos = min(s["dist_m"] for s in sensors)
    if exit_pos is None:
        exit_pos = max(s["dist_m"] for s in sensors)
    lo_pos, hi_pos = min(entry_pos, exit_pos), max(entry_pos, exit_pos)

    sigma_t = det["sigma_t_s"]
    p_d = det["p_d"]
    t_merge = det["t_merge_s"]
    far = det["false_alarm"]["rate_per_station_per_min"]

    vehicles = generate_vehicles(cfg, rng, window_seconds)

    # 1) TRUE crossing events: (time, sensor, vehicle)
    raw_events = []   # ground-truth-caused detections before merge/miss
    gt = {v["vehicle_id"]: {
            "vehicle_id": v["vehicle_id"],
            "entry_time": v["entry_time"],
            "speed": v["speed"],
            "true_crossing_time_per_station": {},
            "caused_detection_ids": [],
            "missed_stations": [],
            "merged_with": set(),
          } for v in vehicles}

    for v in vehicles:
        for s in sensors:
            d = s["dist_m"]
            if not (lo_pos <= d <= hi_pos):
                continue  # sensor not within the span this vehicle travels
            # distance from entry to this sensor along the bridge
            travel = abs(d - entry_pos)
            t_true = v["entry_time"] + travel / v["speed"]
            gt[v["vehicle_id"]]["true_crossing_time_per_station"][s["sensor_id"]] = round(t_true, 4)

            # detection model: miss with prob (1 - p_d)
            if rng.random() > p_d:
                gt[v["vehicle_id"]]["missed_stations"].append(s["sensor_id"])
                continue
            t_obs = t_true + rng.gauss(0.0, sigma_t)
            if t_obs < 0 or t_obs >= window_seconds:
                continue
            raw_events.append({
                "sensor_id": s["sensor_id"],
                "dist_m": d,
                "time": t_obs,
                "vehicle_id": v["vehicle_id"],
                "is_false": False,
            })

    # 2) FALSE spikes (clutter) per sensor, Poisson over the window
    for s in sensors:
        if not (lo_pos <= s["dist_m"] <= hi_pos):
            continue
        expected = far * (window_seconds / 60.0)
        n_false = poisson_sample(rng, expected)
        for _ in range(n_false):
            raw_events.append({
                "sensor_id": s["sensor_id"],
                "dist_m": s["dist_m"],
                "time": rng.uniform(0, window_seconds),
                "vehicle_id": None,
                "is_false": True,
            })

    # 3) MERGE: events at the SAME sensor within t_merge collapse to one.
    detections = merge_events(raw_events, t_merge, gt)

    # assign event_ids and finalize ground-truth links
    for i, e in enumerate(detections):
        e["event_id"] = f"w{window_id}_e{i}"
        for vid in e["source_vehicles"]:
            if vid is not None:
                gt[vid]["caused_detection_ids"].append(e["event_id"])
        if len([x for x in e["source_vehicles"] if x is not None]) > 1:
            real = [x for x in e["source_vehicles"] if x is not None]
            for a in real:
                for b in real:
                    if a != b:
                        gt[a]["merged_with"].add(b)

    return detections, gt, window_seconds


def merge_events(raw, t_merge, gt):
    """Collapse same-sensor events within t_merge into single detections."""
    by_sensor = {}
    for e in raw:
        by_sensor.setdefault(e["sensor_id"], []).append(e)
    out = []
    for sid, evs in by_sensor.items():
        evs.sort(key=lambda x: x["time"])
        cluster = []
        def flush(c):
            if not c:
                return
            times = [x["time"] for x in c]
            out.append({
                "sensor_id": sid,
                "longitudinal_position": c[0]["dist_m"],
                "detection_time": sum(times) / len(times),  # merged timestamp
                "source_vehicles": [x["vehicle_id"] for x in c],
                "n_merged": len(c),
            })
        for e in evs:
            if cluster and (e["time"] - cluster[-1]["time"]) <= t_merge:
                cluster.append(e)
            else:
                flush(cluster)
                cluster = [e]
        flush(cluster)
    out.sort(key=lambda x: (x["detection_time"], x["longitudinal_position"]))
    return out


def poisson_sample(rng, lam):
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_outputs(cfg, detections, gt, window_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    det_name = cfg["output"]["detection_csv"]["filename_pattern"].format(window_id=window_id)
    gt_name = cfg["output"]["ground_truth_csv"]["filename_pattern"].format(window_id=window_id)

    with open(os.path.join(out_dir, det_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "sensor_id", "longitudinal_position",
                    "lateral_id", "detection_time", "event_id"])
        for e in detections:
            w.writerow([window_id, e["sensor_id"],
                        f'{e["longitudinal_position"]:.2f}',
                        "",  # lateral_id empty (no lateral in v1)
                        f'{e["detection_time"]:.4f}', e["event_id"]])

    with open(os.path.join(out_dir, gt_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vehicle_id", "entry_time", "speed",
                    "true_crossing_time_per_station", "caused_detection_ids",
                    "missed_stations", "merged_with"])
        for vid, g in sorted(gt.items()):
            w.writerow([
                g["vehicle_id"],
                f'{g["entry_time"]:.4f}',
                f'{g["speed"]:.3f}',
                json.dumps(g["true_crossing_time_per_station"]),
                json.dumps(g["caused_detection_ids"]),
                json.dumps(g["missed_stations"]),
                json.dumps(sorted(g["merged_with"])),
            ])
    return det_name, gt_name


# ---------------------------------------------------------------------------
# Visualization: detection-stream view (two panels — anonymous vs ground truth)
# ---------------------------------------------------------------------------
def _sensor_order(sensors):
    """Map sensor_id -> row index (0 = entry end), ordered by position."""
    ordered = sorted(sensors, key=lambda s: s["dist_m"])
    return {s["sensor_id"]: i for i, s in enumerate(ordered)}, ordered


def render_matplotlib(detections, gt, sensors, window_id, out_dir,
                      t_start=0.0, t_span=60.0):
    """Static two-panel PNG: top = anonymous detections, bottom = ground truth."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping --viz mpl. "
              "Install with `pip install matplotlib`.")
        return None

    row_of, ordered = _sensor_order(sensors)
    n = len(ordered)
    t_end = t_start + t_span

    # group detections by source vehicle for the truth chains
    by_v = {}
    for e in detections:
        for vid in e["source_vehicles"]:
            if vid is not None:
                by_v.setdefault(vid, []).append(e)

    palette = ["#534AB7", "#1D9E75", "#D85A30", "#D4537E", "#185FA5",
               "#639922", "#BA7517", "#A32D2D", "#0F6E56", "#993C1D"]

    # --- scale layout with sensor count ------------------------------------
    # SWAPPED AXES: sensors on X (categorical, evenly spaced by order),
    # time on Y (increasing UP).
    idx_of = {s["sensor_id"]: i for i, s in enumerate(ordered)}  # even spacing
    # figure width grows with n; height fixed for the time axis
    panel_w = max(6.0, min(0.30 * n, 16.0))
    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(panel_w * 2, 7),
                                         sharey=True)
    # dot size and label density adapt to n
    ms = 4 if n <= 12 else (3 if n <= 25 else 2)
    label_step = max(1, int(math.ceil(n / 20.0)))
    ticks = list(range(n))
    ticklabels = [f"S{i+1}" if (i % label_step == 0) else ""
                  for i in range(n)]

    def setup(ax, title):
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7, rotation=90)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(t_start, t_end)            # time UP (bottom = window start)
        ax.set_xlabel("sensor (ascending distance order)", fontsize=8)
        for i in range(n):
            ax.axvline(i, color="#888780", alpha=0.08, lw=0.6)

    setup(ax_top, "What the tracker sees (anonymous)")
    setup(ax_bot, "Ground truth (colored + connected by true vehicle)")
    ax_top.set_ylabel("time (s)", fontsize=8)

    in_win = lambda e: t_start <= e["detection_time"] <= t_end
    # top: all anonymous  (x = sensor index, y = time)
    for e in detections:
        if not in_win(e):
            continue
        ax_top.plot(idx_of[e["sensor_id"]], e["detection_time"], "o",
                    ms=ms, color="#888780")
    # bottom: chains then colored dots
    for vid, evs in by_v.items():
        evs2 = [e for e in evs if in_win(e)]
        if len(evs2) < 1:
            continue
        evs2.sort(key=lambda e: e["detection_time"])   # order along time
        col = palette[vid % len(palette)]
        xs = [idx_of[e["sensor_id"]] for e in evs2]
        ys = [e["detection_time"] for e in evs2]
        ax_bot.plot(xs, ys, "-", color=col, lw=0.8, alpha=0.8)
    for e in detections:
        if not in_win(e):
            continue
        reals = [v for v in e["source_vehicles"] if v is not None]
        if not reals:
            col = "#E24B4A"            # false
        elif len(reals) > 1:
            col = "#EF9F27"            # merged
        else:
            col = palette[reals[0] % len(palette)]
        ax_bot.plot(idx_of[e["sensor_id"]], e["detection_time"], "o",
                    ms=ms, color=col)

    fig.tight_layout()
    path = os.path.join(out_dir, f"viz_window_{window_id}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_html(det_name, gt_name, sensors, window_id, out_dir,
                window_seconds):
    """Self-contained HTML viewer that loads the actual generated CSVs."""
    _, ordered = _sensor_order(sensors)
    positions = [s["dist_m"] for s in ordered]
    sensor_ids = [s["sensor_id"] for s in ordered]
    meta = {
        "positions": positions,
        "sensor_ids": sensor_ids,
        "window_seconds": window_seconds,
        "det_file": det_name,
        "gt_file": gt_name,
    }
    html = _HTML_TEMPLATE.replace("/*__META__*/", json.dumps(meta))
    path = os.path.join(out_dir, f"viewer_window_{window_id}.html")
    with open(path, "w") as f:
        f.write(html)
    return path


_HTML_TEMPLATE = r"""../traffic_simulation/sim.html"""


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/traffic_simulation/traffic_sim_config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--viz", choices=["none", "html", "mpl", "both"],
                    default="none",
                    help="Generate a detection-stream visualization for each "
                         "window: html viewer, matplotlib png, both, or none.")
    args = ap.parse_args()

     # ── Load bridge model ─────────────────────────────────────────────────────
    print("Loading bridge model...")
    bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    junctions = bridge.find_boundaries()

    boundary_sensors = [s for j in junctions for s in j.sensor_ids()]
    boundary_sensors = list(dict.fromkeys(boundary_sensors))  # deduplicate, preserve order

    print(f"Boundary sensors (after deduplication): {len(boundary_sensors)}")
    #print(boundary_sensors)

    cfg = load_config(args.config)
    sensors = load_sensors(cfg)
    #print(sensors)

    sensors = [d for d in sensors if any(i in boundary_sensors for i in d.values())]

    #print(sensors)
    out_dir = args.out or cfg["output"]["out_dir"]
    window_seconds = cfg["output"]["window_minutes"] * 60
    total_minutes = cfg["output"].get("total_minutes", cfg["output"]["window_minutes"])
    n_windows = max(1, int(round(total_minutes / cfg["output"]["window_minutes"])))

    base_seed = cfg["run"]["global_seed"]
    summary = []
    for w in range(n_windows):
        rng = random.Random(base_seed + w)  # reproducible, distinct per window
        detections, gt, _ = simulate_window(cfg, sensors, rng, w, window_seconds)
        det_name, gt_name = write_outputs(cfg, detections, gt, w, out_dir)
        n_false = sum(1 for e in detections if all(v is None for v in e["source_vehicles"]))
        n_merged = sum(1 for e in detections if e["n_merged"] > 1)
        entry = {
            "window": w, "vehicles": len(gt), "detections": len(detections),
            "false_spikes": n_false, "merged_events": n_merged,
            "det_file": det_name, "gt_file": gt_name,
        }
        if args.viz in ("mpl", "both"):
            png = render_matplotlib(detections, gt, sensors, w, out_dir)
            if png:
                entry["png"] = os.path.basename(png)
        if args.viz in ("html", "both"):
            htmlp = render_html(det_name, gt_name, sensors, w, out_dir,
                                window_seconds)
            entry["html"] = os.path.basename(htmlp)
        summary.append(entry)

    print(json.dumps(summary, indent=2))
    if args.viz in ("html", "both"):
        print("\nTip: open the .html viewer via a local server so it can "
              "auto-load the CSVs:\n  cd %s && python3 -m http.server\n"
              "then browse to the viewer_window_*.html file. Opening the file "
              "directly (file://) also works — use the file pickers to select "
              "the two CSVs." % out_dir)


if __name__ == "__main__":
    main()