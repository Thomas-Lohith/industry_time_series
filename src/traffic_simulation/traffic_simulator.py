"""
Bridge Traffic Simulation — synthetic sensor detection generator.

For each scenario, seed and window it produces:
  - a DETECTION csv    (noisy per-sensor timestamps: what the tracker sees)
  - a GROUND-TRUTH csv (which vehicle caused which detection, plus speed class)
and writes one row per (scenario, seed, window) to <out_dir>/summary.csv.

Design notes:
  * Sensor geometry is read from the sensor CSV (source of truth).
  * Only the vertical channel timestamp is used; amplitude is not modelled.
  * Motion model comes from the config (constant_speed / constant_acceleration /
    maneuver_chain). Vehicles are independent (no car-following, no lanes), so a
    faster vehicle can overtake a slower one freely.
  * Entry speeds come from a mixture of speed classes (traffic.speed_classes).
    The physical clamp while driving is traffic.speed_limits.
  * Scenarios in the config override keys inside `traffic`; choose one with
    --scenario NAME, or run all with --scenario all.
  * Arrivals, speeds, motion and detection use SEPARATE random streams, so for a
    given seed all scenarios see the same arrival times (unless a scenario
    changes the arrival settings).

Every probability / spread in the config is an untuned placeholder.
"""

import argparse
import bisect
import copy
import csv
import json
import math
import os
import random
from statistics import NormalDist

from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter


# ---------------------------------------------------------------------------
# Config loading + scenario overrides
# ---------------------------------------------------------------------------
def load_config(path):
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        raise SystemExit(
            "PyYAML not available or config unreadable. Install with "
            "`pip install pyyaml`."
        )


def _deep_merge(base, override):
    """Merge `override` into `base` in place: dicts merge, everything else replaces."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def apply_scenario(cfg, name):
    """Return a copy of cfg with scenarios[name] merged into cfg['traffic']."""
    scenarios = cfg.get("scenarios") or {}
    if name not in scenarios:
        raise SystemExit(
            f"Unknown scenario {name!r}. Available: {', '.join(scenarios) or '(none)'}"
        )
    out = copy.deepcopy(cfg)
    _deep_merge(out["traffic"], scenarios[name] or {})
    return out


def speed_limits(tr):
    """Physical speed clamp. Falls back to the legacy `speed` block."""
    lim = tr.get("speed_limits") or tr.get("speed")
    if not lim:
        raise ValueError("Config needs traffic.speed_limits (or legacy traffic.speed).")
    return lim["min_mps"], lim["max_mps"]


def speed_classes(tr):
    """Entry-speed classes. Falls back to the legacy single `speed` block."""
    classes = tr.get("speed_classes")
    if not classes:
        sp = tr.get("speed")
        if not sp:
            raise ValueError("Config needs traffic.speed_classes (or legacy traffic.speed).")
        classes = [{"name": "default", "weight": 1.0,
                    "mean_mps": sp["mean_mps"], "std_mps": sp["std_mps"],
                    "lo_mps": sp["min_mps"], "hi_mps": sp["max_mps"]}]
    v_min, v_max = speed_limits(tr)
    for c in classes:
        for key in ("name", "mean_mps", "std_mps", "lo_mps", "hi_mps"):
            if key not in c:
                raise ValueError(f"speed class {c!r} is missing {key!r}")
        if c["lo_mps"] < v_min or c["hi_mps"] > v_max:
            raise ValueError(
                f"speed class {c['name']!r} band [{c['lo_mps']}, {c['hi_mps']}] "
                f"lies outside speed_limits [{v_min}, {v_max}]"
            )
    return classes


# ---------------------------------------------------------------------------
# Random streams
# ---------------------------------------------------------------------------
RNG_STREAMS = ("arrival", "speed", "motion", "detect")


def make_rngs(seed, window_id):
    """Independent, reproducible streams. String seeds are hashed
    deterministically (SHA-512) by random.Random."""
    return {k: random.Random(f"{seed}-{window_id}-{k}") for k in RNG_STREAMS}


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
                "sensor_vertical": row[cols["vertical"]],
                "dist_m": float(row[cols["distance"]]),
                "span": row.get(cols["span"], ""),
            })
    # Order stations by position so detection sequences are positional.
    sensors.sort(key=lambda d: d["dist_m"])
    return sensors


def span_bounds(cfg, sensors):
    """(entry_pos, lo_pos, hi_pos) of the stretch vehicles actually travel."""
    bridge = cfg["bridge"]
    entry_pos = bridge.get("entry_position_m")
    exit_pos = bridge.get("exit_position_m")
    if entry_pos is None:
        entry_pos = min(s["dist_m"] for s in sensors)
    if exit_pos is None:
        exit_pos = max(s["dist_m"] for s in sensors)
    return entry_pos, min(entry_pos, exit_pos), max(entry_pos, exit_pos)


# ---------------------------------------------------------------------------
# Vehicle generation
# ---------------------------------------------------------------------------
def truncated_normal(rng, mean, std, lo, hi):
    """Exact truncated-normal draw via inverse CDF (one RNG draw per call).
    Works correctly even for bands far in the tail."""
    if std <= 0:
        return min(max(mean, lo), hi)
    nd = NormalDist(mean, std)
    a, b = nd.cdf(lo), nd.cdf(hi)
    if b - a < 1e-12:                      # band has ~no probability mass
        return min(max(mean, lo), hi)
    u = min(max(rng.uniform(a, b), 1e-15), 1.0 - 1e-15)
    return min(max(nd.inv_cdf(u), lo), hi)


def crossing_time(entry_time, dist, v0, a, v_min, v_max):
    """Time at which a vehicle that entered at entry_time with speed v0 and
    constant acceleration a has covered `dist` metres.

    Speed is clamped to [v_min, v_max]: once the limit is reached the vehicle
    continues at that constant speed (piecewise model).
    """
    if dist <= 0:
        return entry_time
    if a == 0.0:
        return entry_time + dist / v0
    v_lim = v_max if a > 0 else v_min
    t_lim = (v_lim - v0) / a              # time to reach the speed limit
    d_lim = v0 * t_lim + 0.5 * a * t_lim ** 2
    if dist <= d_lim:
        disc = v0 * v0 + 2.0 * a * dist   # >= 0 here because dist <= d_lim
        t = (-v0 + math.sqrt(max(disc, 0.0))) / a
    else:
        t = t_lim + (dist - d_lim) / v_lim
    return entry_time + t


# ---------------------------------------------------------------------------
# Trajectory builders (per-vehicle speed history along the bridge)
# ---------------------------------------------------------------------------
# A trajectory is a list of segments (x0, t0, v0, a, dur):
#   x0  : distance from entry at segment start (m)
#   t0  : time since entry at segment start (s)
#   v0  : speed at segment start (m/s)
#   a   : constant acceleration within the segment (m/s^2)
#   dur : segment duration (s)
# Speed is continuous across segments. Vehicles do not react to each other.
def _split_at_speed_limit(v, a, dur, v_min, v_max):
    """Split one (a, dur) phase so speed never leaves [v_min, v_max]; after the
    limit is reached the vehicle cruises at that limit for the rest of dur."""
    eps = 1e-9
    if a > 0 and v + a * dur > v_max:
        t1 = (v_max - v) / a
    elif a < 0 and v + a * dur < v_min:
        t1 = (v_min - v) / a
    else:
        return [(dur, a)]
    if t1 <= eps:
        return [(dur, 0.0)]
    return [(t1, a), (dur - t1, 0.0)]


def build_maneuver_trajectory(rng, cfg, v0, length_m):
    """Independent maneuver chain: cruise/accelerate/coast/brake phases chosen by
    a Markov chain whose weights are biased by how far speed is from the
    vehicle's desired speed (its entry speed)."""
    tr = cfg["traffic"]
    man = tr["maneuvers"]
    v_min, v_max = speed_limits(tr)
    phases = man["phases"]
    trans = man["transitions"]
    k = man.get("speed_bias", 0.0)
    v_des = v0

    phase = man.get("initial_phase", "cruise")
    segs = []
    x = t = 0.0
    v = v0
    for _ in range(100000):
        if x >= length_m:
            break
        ph = phases[phase]
        dur = rng.uniform(*ph["duration_s"])
        a = truncated_normal(rng, ph["accel_mean_mps2"], ph["accel_std_mps2"],
                             *ph["accel_range_mps2"])
        for d_seg, a_seg in _split_at_speed_limit(v, a, dur, v_min, v_max):
            if d_seg <= 0:
                continue
            segs.append((x, t, v, a_seg, d_seg))
            x += v * d_seg + 0.5 * a_seg * d_seg ** 2
            t += d_seg
            v = min(max(v + a_seg * d_seg, v_min), v_max)

        # next phase: transition weights, biased by speed deviation
        dev = (v_des - v) / v_des          # >0: below desired speed
        names, weights = [], []
        for nxt, w in trans[phase].items():
            mean_a = phases[nxt]["accel_mean_mps2"]
            if mean_a > 0:
                w *= math.exp(k * dev)
            elif mean_a < 0:
                w *= math.exp(-k * dev)
            names.append(nxt)
            weights.append(w)
        phase = rng.choices(names, weights=weights)[0]
    else:
        raise RuntimeError("maneuver trajectory did not reach the bridge end")
    return segs


TRAJECTORY_BUILDERS = {"maneuver_chain": build_maneuver_trajectory}


def trajectory_crossing_time(entry_time, dist, segs):
    """Exact time at which a vehicle with trajectory `segs` has travelled `dist`."""
    if dist <= 0:
        return entry_time
    x0s = [sg[0] for sg in segs]
    i = max(bisect.bisect_right(x0s, dist) - 1, 0)
    x0, t0, v0, a, _ = segs[i]
    dx = dist - x0
    # stable root of 0.5*a*tau^2 + v0*tau - dx = 0 (also valid for a = 0)
    tau = 2.0 * dx / (v0 + math.sqrt(max(v0 * v0 + 2.0 * a * dx, 0.0)))
    return entry_time + t0 + tau


def trajectory_to_json(segs):
    return [[round(sg[4], 3), round(sg[3], 3), round(sg[2], 3)] for sg in segs]


def generate_vehicles(cfg, rngs, window_seconds):
    """Vehicle arrivals over the window. Each vehicle gets a speed class and an
    entry speed drawn from that class.

    Gap selection is controlled by traffic.arrival.process:
      - "poisson":   exponential inter-arrival gap, rate = rate_veh_per_min
      - "gap_cycle": gaps taken in order from gap_values_s, cycling back to
                     the start once the list is exhausted. Restarts at
                     index 0 for every window.
    """
    rng_arr, rng_spd, rng_mot = rngs["arrival"], rngs["speed"], rngs["motion"]
    tr = cfg["traffic"]
    arr = tr["arrival"]
    process = arr.get("process", "poisson")
    rate_per_s = arr.get("rate_veh_per_min", 0.0) / 60.0
    gap_values = arr.get("gap_values_s")
    use_acc = tr.get("motion_model", "constant_speed") == "constant_acceleration"
    acc = tr.get("acceleration", {})

    classes = speed_classes(tr)
    weights = [float(c.get("weight", 1.0)) for c in classes]
    if sum(weights) <= 0:
        raise ValueError("speed_classes weights must sum to a positive number.")

    vehicles = []
    t = 0.0
    v_id = 0
    gap_idx = 0
    while True:
        if process == "gap_cycle":
            if not gap_values:
                raise ValueError(
                    "traffic.arrival.gap_values_s must be a non-empty list "
                    "when traffic.arrival.process is 'gap_cycle'."
                )
            gap = gap_values[gap_idx % len(gap_values)]
            gap_idx += 1
        elif process == "poisson":
            gap = rng_arr.expovariate(rate_per_s) if rate_per_s > 0 else float("inf")
        else:
            raise ValueError(
                f"Unknown traffic.arrival.process: {process!r} "
                "(expected 'poisson' or 'gap_cycle')."
            )
        t += gap
        if t >= window_seconds:
            break
        cls = rng_spd.choices(classes, weights=weights)[0]
        speed = truncated_normal(rng_spd, cls["mean_mps"], cls["std_mps"],
                                 cls["lo_mps"], cls["hi_mps"])
        a = (rng_mot.gauss(acc.get("mean_mps2", 0.0), acc.get("std_mps2", 0.0))
             if use_acc else 0.0)
        vehicles.append({"vehicle_id": v_id, "entry_time": t, "speed": speed,
                         "acceleration": a, "speed_class": cls["name"]})
        v_id += 1
    return vehicles


# ---------------------------------------------------------------------------
# Core: true crossing times, then apply detection model
# ---------------------------------------------------------------------------
def simulate_window(cfg, sensors, rngs, window_id, window_seconds):
    det = cfg["detection"]
    rng_det = rngs["detect"]
    entry_pos, lo_pos, hi_pos = span_bounds(cfg, sensors)

    sigma_t = det["sigma_t_s"]
    p_d = det.get("p_d", 1.0)          # not in config -> every crossing detected
    t_merge = det["t_merge_s"]
    far = det.get("false_alarm", {}).get("rate_per_station_per_min", 0.0)  # -> no clutter

    vehicles = generate_vehicles(cfg, rngs, window_seconds)
    v_min, v_max = speed_limits(cfg["traffic"])
    motion = cfg["traffic"].get("motion_model", "constant_speed")
    if motion in TRAJECTORY_BUILDERS:
        for v in vehicles:
            v["trajectory"] = TRAJECTORY_BUILDERS[motion](
                rngs["motion"], cfg, v["speed"], (hi_pos - lo_pos) + 1.0)

    # 1) TRUE crossing events: (time, sensor, vehicle)
    raw_events = []
    gt = {v["vehicle_id"]: {
            "vehicle_id": v["vehicle_id"],
            "entry_time": v["entry_time"],
            "speed": v["speed"],
            "speed_class": v["speed_class"],
            "acceleration": None if "trajectory" in v else v["acceleration"],
            "trajectory": trajectory_to_json(v["trajectory"]) if "trajectory" in v else [],
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
            travel = abs(d - entry_pos)
            if "trajectory" in v:
                t_true = trajectory_crossing_time(v["entry_time"], travel, v["trajectory"])
            else:
                t_true = crossing_time(v["entry_time"], travel, v["speed"],
                                       v["acceleration"], v_min, v_max)
            gt[v["vehicle_id"]]["true_crossing_time_per_station"][s["sensor_id"]] = round(t_true, 4)

            # detection model: miss with prob (1 - p_d)
            if rng_det.random() > p_d:
                gt[v["vehicle_id"]]["missed_stations"].append(s["sensor_id"])
                continue
            t_obs = t_true + rng_det.gauss(0.0, sigma_t)
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
        n_false = poisson_sample(rng_det, expected)
        for _ in range(n_false):
            raw_events.append({
                "sensor_id": s["sensor_id"],
                "dist_m": s["dist_m"],
                "time": rng_det.uniform(0, window_seconds),
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
        real = [x for x in e["source_vehicles"] if x is not None]
        if len(real) > 1:
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
# Scenario diagnostics
# ---------------------------------------------------------------------------
def count_overtakes(gt, first_sid, last_sid):
    """Number of vehicle pairs whose order at the last in-span sensor differs
    from their order at the first in-span sensor (true times, no noise).
    Direction-agnostic. O(n^2), fine for a few hundred vehicles."""
    pts = []
    for g in gt.values():
        tc = g["true_crossing_time_per_station"]
        if first_sid in tc and last_sid in tc:
            pts.append((tc[first_sid], tc[last_sid]))
    n = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (pts[i][0] - pts[j][0]) * (pts[i][1] - pts[j][1]) < 0:
                n += 1
    return n


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
                        "",  # lateral_id empty (no lateral grouping)
                        f'{e["detection_time"]:.4f}', e["event_id"]])

    with open(os.path.join(out_dir, gt_name), "w", newline="") as f:
        w = csv.writer(f)
        # speed_class is appended LAST so existing column positions are unchanged
        w.writerow(["vehicle_id", "entry_time", "speed", "acceleration", "trajectory",
                    "true_crossing_time_per_station", "caused_detection_ids",
                    "missed_stations", "merged_with", "speed_class"])
        for vid, g in sorted(gt.items()):
            w.writerow([
                g["vehicle_id"],
                f'{g["entry_time"]:.4f}',
                f'{g["speed"]:.3f}',
                "" if g["acceleration"] is None else f'{g["acceleration"]:.4f}',
                json.dumps(g["trajectory"]),
                json.dumps(g["true_crossing_time_per_station"]),
                json.dumps(g["caused_detection_ids"]),
                json.dumps(g["missed_stations"]),
                json.dumps(sorted(g["merged_with"])),
                g["speed_class"],
            ])
    return det_name, gt_name


SUMMARY_FIELDS = ["scenario", "seed", "window", "vehicles", "vehicles_per_class",
                  "detections", "merged_events", "false_spikes", "overtakes",
                  "out_dir", "det_file", "gt_file"]


def update_summary(path, rows):
    """Write rows to summary.csv, replacing any existing rows with the same
    (scenario, seed, window). Extra columns already in the file (e.g. precision /
    recall added by the tracking script) are preserved for untouched rows."""
    existing, fields = [], list(SUMMARY_FIELDS)
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            for name in reader.fieldnames or []:
                if name not in fields:
                    fields.append(name)

    def key(r):
        return (str(r["scenario"]), str(r["seed"]), str(r["window"]))

    new_keys = {key(r) for r in rows}
    kept = [r for r in existing if key(r) not in new_keys]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        for r in kept + rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Visualization: detection-stream view (two panels — anonymous vs ground truth)
# ---------------------------------------------------------------------------
def _sensor_order(sensors):
    """Map sensor_id -> row index (0 = entry end), ordered by position."""
    ordered = sorted(sensors, key=lambda s: s["dist_m"])
    return {s["sensor_id"]: i for i, s in enumerate(ordered)}, ordered


def render_matplotlib(detections, gt, sensors, window_id, out_dir,
                      t_start=0.0, t_span=60.0):
    """Static two-panel PNG: left = anonymous detections, right = ground truth."""
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

    by_v = {}
    for e in detections:
        for vid in e["source_vehicles"]:
            if vid is not None:
                by_v.setdefault(vid, []).append(e)

    palette = ["#534AB7", "#1D9E75", "#D85A30", "#D4537E", "#185FA5",
               "#639922", "#BA7517", "#A32D2D", "#0F6E56", "#993C1D"]

    idx_of = {s["sensor_id"]: i for i, s in enumerate(ordered)}
    panel_w = max(6.0, min(0.30 * n, 16.0))
    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(panel_w * 2, 7),
                                         sharey=True)
    ms = 4 if n <= 12 else (3 if n <= 25 else 2)
    label_step = max(1, int(math.ceil(n / 20.0)))
    ticks = list(range(n))
    ticklabels = [f"S{i+1}" if (i % label_step == 0) else "" for i in range(n)]

    def setup(ax, title):
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7, rotation=90)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(t_start, t_end)
        ax.set_xlabel("sensor (ascending distance order)", fontsize=8)
        for i in range(n):
            ax.axvline(i, color="#888780", alpha=0.08, lw=0.6)

    setup(ax_top, "What the tracker sees (anonymous)")
    setup(ax_bot, "Ground truth (colored + connected by true vehicle)")
    ax_top.set_ylabel("time (s)", fontsize=8)

    in_win = lambda e: t_start <= e["detection_time"] <= t_end
    for e in detections:
        if not in_win(e):
            continue
        ax_top.plot(idx_of[e["sensor_id"]], e["detection_time"], "o",
                    ms=ms, color="#888780")
    for vid, evs in by_v.items():
        evs2 = [e for e in evs if in_win(e)]
        if len(evs2) < 1:
            continue
        # Connect in sensor (position) order, not detection-time order.
        evs2.sort(key=lambda e: (idx_of[e["sensor_id"]], e["detection_time"]))
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


# Viewer template lives next to this script; the marker is replaced by the
# window metadata (see sim.html).
_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim.html")
_META_MARKER = "/*__META__*/{}"


def render_html(det_name, gt_name, sensors, window_id, out_dir, window_seconds):
    """Self-contained HTML viewer that loads the generated CSVs."""
    if not os.path.exists(_TEMPLATE_PATH):
        print(f"HTML template not found at {_TEMPLATE_PATH}; skipping --viz html.")
        return None
    _, ordered = _sensor_order(sensors)
    meta = {
        "positions": [s["dist_m"] for s in ordered],
        "sensor_ids": [s["sensor_id"] for s in ordered],
        "window_seconds": window_seconds,
        "det_file": det_name,
        "gt_file": gt_name,
    }
    with open(_TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    if _META_MARKER not in template:
        raise ValueError(f"{_TEMPLATE_PATH} has no {_META_MARKER!r} marker")
    html = template.replace(_META_MARKER, json.dumps(meta), 1)
    path = os.path.join(out_dir, f"viewer_window_{window_id}.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/Users/thomas/Desktop/github_repos/industry_time_series/src/config/traffic_sim_config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario", default=None,
                    help="Scenario name from the config, or 'all'. "
                         "Default: the config's `scenario` key.")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="Override run.n_seeds from the config.")
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
    boundary_sensors = list(dict.fromkeys(boundary_sensors))  # dedupe, keep order
    print(f"Boundary sensors (after deduplication): {len(boundary_sensors)}")

    cfg = load_config(args.config)
    sensors = load_sensors(cfg)

    # Keep only boundary sensors. Matches on sensor_id OR the vertical-channel
    # id (the original filter matched any field), compared as strings.
    keep = {str(x) for x in boundary_sensors}
    sensors = [s for s in sensors
               if str(s["sensor_id"]) in keep or str(s["sensor_vertical"]) in keep]
    if not sensors:
        raise SystemExit("No sensors left after the boundary filter. Check that "
                         "the IDs in the sensor CSV match the bridge model IDs.")
    print(f"Sensors used: {len(sensors)}")

    base_out = args.out or cfg["output"]["out_dir"]
    window_seconds = cfg["output"]["window_minutes"] * 60
    total_minutes = cfg["output"].get("total_minutes", cfg["output"]["window_minutes"])
    n_windows = max(1, int(round(total_minutes / cfg["output"]["window_minutes"])))

    base_seed = cfg["run"]["global_seed"]
    n_seeds = args.n_seeds or cfg["run"].get("n_seeds", 1)
    seeds = list(range(base_seed, base_seed + n_seeds))

    requested = args.scenario or cfg.get("scenario")
    if requested == "all":
        names = list((cfg.get("scenarios") or {}).keys())
        if not names:
            raise SystemExit("--scenario all, but the config has no scenarios section.")
    elif requested:
        names = [requested]
    else:
        names = [None]                       # no scenarios: run the base config

    rows = []
    for name in names:
        scfg = apply_scenario(cfg, name) if name else cfg
        label = name or "default"

        _, lo_pos, hi_pos = span_bounds(scfg, sensors)
        in_span = [s for s in sensors if lo_pos <= s["dist_m"] <= hi_pos]
        first_sid = in_span[0]["sensor_id"] if in_span else None
        last_sid = in_span[-1]["sensor_id"] if in_span else None

        for seed in seeds:
            out_dir = os.path.join(base_out, label, f"seed_{seed}")
            for w in range(n_windows):
                rngs = make_rngs(seed, w)
                detections, gt, _ = simulate_window(scfg, sensors, rngs, w, window_seconds)
                det_name, gt_name = write_outputs(scfg, detections, gt, w, out_dir)

                class_counts = {}
                for g in gt.values():
                    class_counts[g["speed_class"]] = class_counts.get(g["speed_class"], 0) + 1
                overtakes = count_overtakes(gt, first_sid, last_sid) if first_sid else 0

                row = {
                    "scenario": label, "seed": seed, "window": w,
                    "vehicles": len(gt),
                    "vehicles_per_class": json.dumps(class_counts, sort_keys=True),
                    "detections": len(detections),
                    "merged_events": sum(1 for e in detections if e["n_merged"] > 1),
                    "false_spikes": sum(1 for e in detections
                                        if all(v is None for v in e["source_vehicles"])),
                    "overtakes": overtakes,
                    "out_dir": out_dir, "det_file": det_name, "gt_file": gt_name,
                }
                rows.append(row)
                print(f"[{label} | seed {seed} | window {w}] "
                      f"vehicles={row['vehicles']} detections={row['detections']} "
                      f"merged={row['merged_events']} overtakes={overtakes} "
                      f"classes={row['vehicles_per_class']}")

                if args.viz in ("mpl", "both"):
                    render_matplotlib(detections, gt, sensors, w, out_dir)
                if args.viz in ("html", "both"):
                    render_html(det_name, gt_name, sensors, w, out_dir, window_seconds)

    summary_path = os.path.join(base_out, "summary.csv")
    update_summary(summary_path, rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()