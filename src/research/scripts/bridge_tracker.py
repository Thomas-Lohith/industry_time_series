"""
bridge_tracker.py  —  ROLLING ANCHOR FORWARD PROJECTION
========================================================
At every confirmed sensor match, project expected arrival windows
to ALL remaining sensors from that new anchor.

On a miss, anchor stays at last confirmed match — downstream
windows are still valid and computed from the same stable reference.

Matching rule  : event timestamp falls inside [t_min - margin, t_max + margin]
Anchor update  : ONLY on confirmed match (miss never moves the anchor)
After each match: ALL remaining sensor windows are recomputed and displayed

Input sources
-------------
sensor positions  : boundary sensors only, via load_bridge() + find_boundaries()
sensor_events.csv : wide format, one row per vehicle
wim_entries.csv   : StartTimeStr, Velocity (km/h), GrossWeight (kg)

Paths from config.py
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


from src.shared import config
from src.shared.bridge_model import *

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

VELOCITY_TOLERANCE_KMH = 10.0    # ±km/h applied to WIM speed → v_min / v_max
TIMING_MARGIN_MS       = 30000   # ms added each side of every window
TIMESTAMP_FORMAT       = "%Y-%m-%d %H:%M:%S"
_SENSOR_COL_SUFFIX     = "_dominant_peaks"

# Physical position of the WIM gate in bridge model coordinates (metres).
# Derived from two real vehicle observations:
#   v1 actual speed = (238.9 - 99.8) / (154.56 - 148.87) = 24.45 m/s (88.0 km/h)
#   WIM gate pos    = 99.8 - 24.45 × 148.87             = -3,540m
#   v2 cross-check  = (99.8 - (-3540)) / 153.58          = 23.70 m/s (85.3 km/h) ✓
WIM_GATE_POSITION_M    = -3540.0

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensorEvent:
    sensor_id      : str
    timestamp      : float
    peak_amplitude : float


@dataclass
class MatchRecord:
    """Result of one track–sensor interaction."""
    sensor_id  : str
    sensor_pos : float
    t_min      : float          # window start (after margin applied)
    t_max      : float          # window end   (after margin applied)
    mu         : float          # window midpoint (no margin)
    matched    : bool
    event_ts   : Optional[float] = None
    event_amp  : Optional[float] = None
    time_error : Optional[float] = None   # event_ts - mu (signed, seconds)


@dataclass
class Vehicle:
    """
    anchor_pos / anchor_ts — position and timestamp of last CONFIRMED match.
    Only updated on a match. A miss leaves the anchor unchanged so all
    downstream windows continue to project from the last good fix.
    """
    vehicle_id   : int
    entry_ts   : float
    v_min      : float
    v_max      : float
    mass_kg    : float
    v_est      : float
    anchor_pos : float          # starts at WIM gate position
    anchor_ts  : float          # starts at WIM entry timestamp
    log        : list = field(default_factory=list)
    status     : str  = "ACTIVE"

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_reference_epoch: Optional[datetime] = None


def _dt_to_seconds(dt: datetime) -> float:
    """Convert datetime to seconds from reference epoch. Sets epoch on first call."""
    global _reference_epoch
    if _reference_epoch is None:
        _reference_epoch = dt
    return (dt - _reference_epoch).total_seconds()


def _to_ts(seconds: float) -> str:
    """
    Convert internal seconds value back to a human-readable timestamp string.
    Format: HH:MM:SS.mmm  (millisecond precision)
    Falls back to raw seconds if epoch not yet set.
    """
    if _reference_epoch is None:
        return f"{seconds:.3f}s"
    dt = _reference_epoch + timedelta(seconds=seconds)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _parse_str_to_dt(ts_str: str) -> datetime:
    """
    Handles two date formats observed in real data:
      YYYY-MM-DD  standard   e.g. 2025-03-18 00:33:00
      YYYY-DD-MM  swapped    e.g. 2025-18-03 00:15:00
    Both represent the same date. Standard tried first, swapped as fallback.
    """
    for fmt in (
        TIMESTAMP_FORMAT + ".%f",
        TIMESTAMP_FORMAT,
        "%Y-%d-%m %H:%M:%S.%f",
        "%Y-%d-%m %H:%M:%S",
    ):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse timestamp: '{ts_str}' "
        f"— expected YYYY-MM-DD or YYYY-DD-MM"
    )


def _parse_timestamp(ts_str: str) -> float:
    return _dt_to_seconds(_parse_str_to_dt(ts_str))

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_sensor_positions() -> dict[str, float]:
    """Boundary sensors only — tail + entry at every span junction."""
    bridge    = load_bridge(config.position_csv, config.threshold_csv)
    junctions = bridge.find_boundaries()

    positions = {}
    for junction in junctions[:2]:
        for sensor in junction.all_sensors():
            positions[sensor.sensor_id] = sensor.distance

    print(f"[LOAD] {len(junctions[:2])} span junctions")
    print(f"[LOAD] {len(positions)} boundary sensors")
    return positions


def load_wim_entries() -> list[Vehicle]:
    """
    MUST be called before load_sensor_events() — sets reference epoch.
    vehicle_id : sequential 0, 1, 2, ... (CSV index ignored)
    anchor   : starts at WIM gate (WIM_GATE_POSITION_M, entry_ts)
    """
    vehicles = []
    with open(config.WIM_CSV, newline="", encoding="utf-8-sig") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            if not row.get("Velocity"):
                continue
            speed_kmh = float(row["Velocity"])
            entry_ts  = _parse_timestamp(row["StartTimeStr"])
            vehicles.append(Vehicle(
                vehicle_id   = idx,
                entry_ts   = entry_ts,
                v_min      = (speed_kmh - VELOCITY_TOLERANCE_KMH) / 3.6,
                v_max      = (speed_kmh + VELOCITY_TOLERANCE_KMH) / 3.6,
                mass_kg    = float(row["GrossWeight"]),
                v_est      = speed_kmh / 3.6,
                anchor_pos = WIM_GATE_POSITION_M,
                anchor_ts  = entry_ts,
            ))
            print(f"  [WIM] Track {idx}: "
                  f"entry={_to_ts(entry_ts)}  "
                  f"speed={speed_kmh:.0f} km/h  "
                  f"v_min={vehicles[-1].v_min:.2f} m/s  "
                  f"v_max={vehicles[-1].v_max:.2f} m/s  "
                  f"mass={vehicles[-1].mass_kg:.0f} kg")

    print(f"[LOAD] {len(vehicles)} WIM entries  (vehicle_ids 0–{len(vehicles)-1})")
    return vehicles


def load_sensor_events() -> dict[str, list[SensorEvent]]:
    """
    Wide-format CSV, one row per vehicle, one column per sensor.
    Date correction: sensor event dates replaced with date from original_timestamp.
    """
    events:  dict[str, list[SensorEvent]] = {}
    seen:    dict[str, set[float]] = {}   # {sensor_id: set of timestamps already loaded}
    skipped = 0
    duplicates = 0

    with open(config.SENSOR_EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        reader      = csv.DictReader(f)
        sensor_cols = [c for c in reader.fieldnames
                       if c.endswith(_SENSOR_COL_SUFFIX)]
        print(f"[LOAD] {len(sensor_cols)} sensor columns in events CSV")

        for row in reader:
            correct_date = _parse_str_to_dt(
                row["original_timestamp"].strip()
            ).date()

            for col in sensor_cols:
                sensor_id = col[: -len(_SENSOR_COL_SUFFIX)]
                cell      = (row[col] or "").strip()
                if not cell:
                    continue
                for entry in cell.split(";"):
                    entry = entry.strip()
                    if not entry or "|" not in entry:
                        continue
                    ts_str, amp_str = entry.split("|", 1)
                    try:
                        event_dt = _parse_str_to_dt(ts_str)
                    except ValueError:
                        skipped += 1
                        continue
                    corrected_dt = event_dt.replace(
                        year=correct_date.year,
                        month=correct_date.month,
                        day=correct_date.day,
                    )
                    ts = _dt_to_seconds(corrected_dt)

                    # Skip if this timestamp already loaded for this sensor
                    # — same physical event appears in multiple vehicle rows
                    # when collection windows overlap
                    if ts in seen.get(sensor_id, set()):
                        duplicates += 1
                        continue

                    seen.setdefault(sensor_id, set()).add(ts)
                    events.setdefault(sensor_id, []).append(SensorEvent(
                        sensor_id      = sensor_id,
                        timestamp      = ts,
                        peak_amplitude = float(amp_str),
                    ))

    total = sum(len(v) for v in events.values())
    if skipped:
        print(f"[LOAD] WARNING: {skipped} entries skipped (unparseable)")
    if duplicates:
        print(f"[LOAD] {duplicates} duplicate events removed (overlapping vehicle windows)")
    print(f"[LOAD] {total} unique events across {len(events)} sensors")
    return events

# ---------------------------------------------------------------------------
# Core — window calculation and matching
# ---------------------------------------------------------------------------

def _compute_window(vehicle: Vehicle,
                    sensor_pos: float) -> tuple[float, float, float]:
    """
    Compute expected arrival window at sensor_pos from current anchor.
    Returns (t_min_with_margin, t_max_with_margin, mu)
    mu = physics midpoint without margin — used for best candidate selection.
    """
    d      = sensor_pos - vehicle.anchor_pos
    t_min  = vehicle.anchor_ts + d / vehicle.v_max
    t_max  = vehicle.anchor_ts + d / vehicle.v_min
    mu     = (t_min + t_max) / 2
    margin = TIMING_MARGIN_MS / 1000.0
    return t_min - margin, t_max + margin, mu


def _attempt_match(vehicle:    Vehicle,
                   sensor_id:  str,
                   sensor_pos: float,
                   events:     list[SensorEvent]) -> MatchRecord:
    """
    Try to match track to events at this sensor using time window.
    Candidates : events inside [t_min, t_max] (margin included)
    Winner     : candidate closest to mu (physics midpoint)
    Anchor     : advances to winner on match, UNCHANGED on miss
    """
    t_min, t_max, mu = _compute_window(vehicle, sensor_pos)
    candidates = [e for e in events if t_min <= e.timestamp <= t_max]

    if not candidates:
        return MatchRecord(
            sensor_id  = sensor_id,
            sensor_pos = sensor_pos,
            t_min      = t_min,
            t_max      = t_max,
            mu         = mu,
            matched    = False,
        )

    # Pick event closest to physics midpoint
    best = min(candidates, key=lambda e: abs(e.timestamp - mu))

    # Refine velocity using travel time from anchor
    d       = sensor_pos - vehicle.anchor_pos
    delta_t = best.timestamp - vehicle.anchor_ts
    if delta_t > 0:
        v_measured  = d / delta_t
        v_measured  = max(vehicle.v_min, min(vehicle.v_max, v_measured))
        vehicle.v_est = 0.5 * v_measured + 0.5 * vehicle.v_est
        vehicle.v_est = max(vehicle.v_min, min(vehicle.v_max, vehicle.v_est))

    # Advance anchor — next projections start from here
    vehicle.anchor_pos = sensor_pos
    vehicle.anchor_ts  = best.timestamp

    return MatchRecord(
        sensor_id  = sensor_id,
        sensor_pos = sensor_pos,
        t_min      = t_min,
        t_max      = t_max,
        mu         = mu,
        matched    = True,
        event_ts   = best.timestamp,
        event_amp  = best.peak_amplitude,
        time_error = best.timestamp - mu,
    )

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_tracker(sensor_positions: dict[str, float],
                sensor_events:    dict[str, list[SensorEvent]],
                vehicles:         list[Vehicle]) -> list[Vehicle]:
    """
    Outer loop: one vehicle at a time.
    Inner loop: sensors in position order.
    After every match, project remaining sensor windows from new anchor.
    """
    ordered = sorted(sensor_positions.items(), key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("BRIDGE TRACKER  —  rolling anchor forward projection")
    print(f"  Boundary sensors  : {len(ordered)}")
    print(f"  WIM tracks        : {len(vehicles)}")
    print(f"  Sensor range      : {ordered[0][1]:.1f}m — {ordered[-1][1]:.1f}m")
    print(f"  WIM gate pos      : {WIM_GATE_POSITION_M:.1f}m")
    print(f"  Timing margin     : ±{TIMING_MARGIN_MS}ms")
    print("=" * 70)

    for vehicle in vehicles[:]:
        print(f"\n{'─' * 70}")
        print(f"TRACK {vehicle.vehicle_id}  |  "
              f"speed={vehicle.v_est * 3.6:.1f} km/h  |  "
              f"v_min={vehicle.v_min:.2f} m/s  v_max={vehicle.v_max:.2f} m/s  |  "
              f"mass={vehicle.mass_kg:.0f} kg")
        print(f"  Anchor start: pos={vehicle.anchor_pos:.1f}m  "
              f"ts={_to_ts(vehicle.anchor_ts)}")
        print(f"{'─' * 70}")

        for i, (sensor_id, sensor_pos) in enumerate(ordered):
            events = sensor_events.get(sensor_id, [])
            rec    = _attempt_match(vehicle, sensor_id, sensor_pos, events)
            vehicle.log.append(rec)

            if rec.matched:
                print(f"\n  ✅ Sensor {sensor_id}  pos={sensor_pos:.1f}m")
                print(f"     window : [{_to_ts(rec.t_min)}, {_to_ts(rec.t_max)}]  "
                      f"width={(rec.t_max - rec.t_min)*1000:.0f}ms")
                print(f"     event  : {_to_ts(rec.event_ts)}  "
                      f"error={rec.time_error * 1000:+.0f}ms  "
                      f"amp={rec.event_amp:+.5f}")
                print(f"     v_est refined → {vehicle.v_est * 3.6:.2f} km/h")
                print(f"     new anchor: pos={vehicle.anchor_pos:.1f}m  "
                      f"ts={_to_ts(vehicle.anchor_ts)}")

                # Project ALL remaining sensors from this new anchor
                remaining = ordered[i + 1:]
                if remaining:
                    print(f"\n     ┌─ Projections from new anchor ({'─' * 30})")
                    for rem_id, rem_pos in remaining[-2:]:
                        rt_min, rt_max, rmu = _compute_window(vehicle, rem_pos)
                        width_ms = (rt_max - rt_min) * 1000
                        print(f"     │  {rem_id:<22}  "
                              f"pos={rem_pos:7.1f}m  "
                              f"window=[{_to_ts(rt_min)}, {_to_ts(rt_max)}]  "
                              f"width={width_ms:.0f}ms")
                    print(f"     └{'─' * 51}")

            else:
                print(f"  ❌ Sensor {sensor_id}  pos={sensor_pos:.1f}m  "
                      f"window=[{_to_ts(rec.t_min)}, {_to_ts(rec.t_max)}]  "
                      f"width={(rec.t_max - rec.t_min)*1000:.0f}ms  "
                      f"events_at_sensor={len(events)}  no match")

        vehicle.status = "COMPLETED"

    return vehicles

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(vehicles: list[Vehicle]) -> None:
    print("\n" + "=" * 70)
    print("TRACK SUMMARY")
    print("=" * 70)

    for vehicle in vehicles:
        matches  = sum(1 for r in vehicle.log if r.matched)
        misses   = sum(1 for r in vehicle.log if not r.matched)
        errors   = [r.time_error * 1000 for r in vehicle.log if r.matched]
        mean_err = sum(errors) / len(errors) if errors else 0.0
        max_err  = max(abs(e) for e in errors) if errors else 0.0

        print(f"\nTrack {vehicle.vehicle_id:>3}  "
              f"v_final={vehicle.v_est * 3.6:.1f} km/h  "
              f"mass={vehicle.mass_kg:.0f} kg  "
              f"matched={matches}/{len(vehicle.log)}  "
              f"missed={misses}  "
              f"mean_err={mean_err:+.0f}ms  "
              f"max_err={max_err:.0f}ms")

        for rec in vehicle.log:
            if rec.matched:
                print(f"  ✅ {rec.sensor_id:<22}  "
                      f"pos={rec.sensor_pos:7.1f}m  "
                      f"event={_to_ts(rec.event_ts)}  "
                      f"error={rec.time_error * 1000:+.0f}ms")
            else:
                print(f"  ❌ {rec.sensor_id:<22}  "
                      f"pos={rec.sensor_pos:7.1f}m  "
                      f"window=[{_to_ts(rec.t_min)}, {_to_ts(rec.t_max)}]")

# ---------------------------------------------------------------------------
# Event timeline with per-track Gaussian probability scores
# ---------------------------------------------------------------------------

def event_timeline_with_probabilities(
    sensor_positions : dict[str, float],
    sensor_events    : dict[str, list[SensorEvent]],
    vehicles         : list[Vehicle],
) -> None:
    """
    For each sensor (in position order), print a timeline of ALL dominant
    peaks recorded at that sensor, and below each event print a Gaussian
    probability score for every vehicle track.

    Probability formula (no normalisation):
        mu    = physics window midpoint  = (t_min + t_max) / 2
        sigma = half physics window width = (t_max - t_min) / 2
        score = exp(-0.5 × ((event.ts - mu) / sigma)²)

        score = 1.0  when event.ts == mu  (dead centre of window)
        score → 0.0  as event.ts moves toward or past window edges

    Margin is NOT included in mu/sigma — scoring uses pure physics only.
    Margin is only used in _attempt_match for binary match/miss decisions.

    Structure designed to extend to all sensors once first-sensor tests pass:
    change  `ordered[:1]`  to  `ordered`  in the sensor loop below.
    """

    ordered = sorted(sensor_positions.items(), key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("EVENT TIMELINE WITH PROBABILITY SCORES")
    print(f"  Sensors shown     : first sensor only (extend ordered[:1] → ordered)")
    print(f"  Tracks            : {len(vehicles)}")
    print(f"  Scoring           : Gaussian(event.ts, mu, sigma)  — no normalisation")
    print("=" * 70)

    # ── change ordered[:1] → ordered to cover all sensors ──────────────────
    for sensor_id, sensor_pos in ordered[:1]:

        events = sensor_events.get(sensor_id, [])

        print(f"\nSensor {sensor_id}  pos={sensor_pos:.1f}m  "
              f"total events={len(events)}")
        print("─" * 70)

        if not events:
            print("  (no events recorded at this sensor)")
            continue

        # Sort all events at this sensor by timestamp for clean timeline view
        timeline = sorted(events, key=lambda e: e.timestamp)

        for event in timeline:

            print(f"\n  Event  {_to_ts(event.timestamp)}  "
                  f"amp={event.peak_amplitude:+.5f}")

            # Score this event against every vehicle independently.
            # Each vehicle's window is computed fresh from its WIM gate
            # anchor (entry_ts, WIM_GATE_POSITION_M) — no dependency on
            # post-matching anchor state from run_tracker.
            for vehicle in vehicles:

                # Distance from WIM gate to this sensor — same geometry
                # for all vehicles, differs only in speed and entry time
                d = sensor_pos - WIM_GATE_POSITION_M

                # Expected arrival window from WIM gate
                # t_min: fastest possible (v_max), t_max: slowest (v_min)
                t_min = vehicle.entry_ts + d / vehicle.v_max
                t_max = vehicle.entry_ts + d / vehicle.v_min

                # Ensure correct order (safety guard)
                t_min, t_max = min(t_min, t_max), max(t_min, t_max)

                mu    = (t_min + t_max) / 2
                sigma = (t_max - t_min) / 2
                if sigma <= 0:
                    sigma = 1e-3

                # Gaussian score — 1.0 at mu, decays toward edges
                score = math.exp(
                    -0.5 * ((event.timestamp - mu) / sigma) ** 2
                )

                print(f"    Vehicle {vehicle.vehicle_id}  "
                      f"entry={_to_ts(vehicle.entry_ts)}  "
                      f"mu={_to_ts(mu)}  "
                      f"window=[{_to_ts(t_min)}, {_to_ts(t_max)}]  "
                      f"sigma={sigma:.3f}s  "
                      f"score={score:.6f}")

        print()


# ---------------------------------------------------------------------------
# Shared helper — build sensor_data dict used by all visualisations
# ---------------------------------------------------------------------------

def _build_sensor_data(
    sensor_positions : dict[str, float],
    sensor_events    : dict[str, list[SensorEvent]],
    vehicles         : list[Vehicle],
) -> dict:
    """
    Pre-compute Gaussian scores for every (sensor, event, vehicle) triple.
    Returns sensor_data dict keyed by sensor_id:
        pos      : float               sensor position (m)
        timeline : [SensorEvent]       events sorted by timestamp
        scores   : [[float]]           scores[event_idx][vehicle_idx]
        mus      : [(vid, t_min, t_max, mu, sigma)]  per vehicle
    """
    ordered     = sorted(sensor_positions.items(), key=lambda x: x[1])
    sensor_data = {}

    for sensor_id, sensor_pos in ordered:
        events = sensor_events.get(sensor_id, [])
        if not events:
            continue

        timeline = sorted(events, key=lambda e: e.timestamp)
        d        = sensor_pos - WIM_GATE_POSITION_M
        mus      = []

        for vehicle in vehicles:
            t_min = vehicle.entry_ts + d / vehicle.v_max
            t_max = vehicle.entry_ts + d / vehicle.v_min
            t_min, t_max = min(t_min, t_max), max(t_min, t_max)
            mu    = (t_min + t_max) / 2
            sigma = (t_max - t_min) / 2 or 1e-3
            mus.append((vehicle.vehicle_id, t_min, t_max, mu, sigma))

        scores = []
        for event in timeline:
            row = []
            for vehicle in vehicles:
                _, t_min, t_max, mu, sigma = mus[vehicle.vehicle_id]
                sc = math.exp(-0.5 * ((event.timestamp - mu) / sigma) ** 2)
                row.append(sc)
            scores.append(row)

        sensor_data[sensor_id] = {
            "pos"      : sensor_pos,
            "timeline" : timeline,
            "scores"   : scores,
            "mus"      : mus,
        }

    return sensor_data


# ---------------------------------------------------------------------------
# Suggestion 1 — Probability Heatmap Matrix
# ---------------------------------------------------------------------------

def viz_heatmap(sensor_data: dict, vehicles: list[Vehicle]) -> None:
    """
    One heatmap per sensor.
    Rows = vehicles, columns = events sorted by timestamp.
    Cell colour = Gaussian probability score (white=0, dark red=1).
    Hot column = event clearly owned by one vehicle.
    Two hot cells in one column = ambiguous event.
    """
    
    sensors = list(sensor_data.keys())
    n_sens  = len(sensors)
    if n_sens == 0:
        print("[VIZ1] No data"); return

    n_cols  = max(len(sensor_data[s]["timeline"]) for s in sensors)
    n_rows  = len(vehicles)

    fig, axes = plt.subplots(
        n_sens, 1,
        figsize=(max(10, n_cols * 1.2), n_sens * (n_rows * 0.55 + 1.2)),
        squeeze=False
    )
    fig.suptitle("Probability Heatmap per Sensor",
                 fontsize=13, fontweight="bold")

    cmap = plt.cm.YlOrRd

    for si, sensor_id in enumerate(sensors):
        data      = sensor_data[sensor_id]
        timeline  = data["timeline"]
        scores    = data["scores"]
        ax        = axes[si][0]
        n_events  = len(timeline)

        matrix = np.array(scores).T    # shape: (n_vehicles, n_events)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       vmin=0, vmax=1, interpolation="nearest")

        # Annotate each cell with score value
        for vi in range(n_rows):
            for ei in range(n_events):
                val = matrix[vi, ei]
                ax.text(ei, vi, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if val > 0.6 else "black")

        # Y axis — vehicle labels
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(
            [f"V{v.vehicle_id}  {v.v_est*3.6:.0f}km/h  {_to_ts(v.entry_ts)}"
             for v in vehicles],
            fontsize=7
        )

        # X axis — event timestamps
        ax.set_xticks(range(n_events))
        ax.set_xticklabels(
            [_to_ts(e.timestamp) for e in timeline],
            rotation=35, ha="right", fontsize=7
        )

        ax.set_title(
            f"Sensor {sensor_id}  pos={data['pos']:.1f}m  "
            f"({n_events} events)",
            fontsize=9, pad=4
        )
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label(
            "Score", fontsize=7)

    plt.tight_layout()
    plt.savefig("viz_options/viz1_heatmap.png", dpi=150, bbox_inches="tight")
    print("[VIZ1] Saved: viz1_heatmap.png")
    plt.show()


# ---------------------------------------------------------------------------
# Suggestion 2 — Vehicle Swimlane Timeline
# ---------------------------------------------------------------------------

def viz_swimlane(
    sensor_data      : dict,
    sensor_positions : dict[str, float],
    vehicles         : list[Vehicle],
) -> None:
    """
    One horizontal lane per vehicle.
    X = timestamp.  Y = lane (one per vehicle).
    Dots = events matched to this vehicle (sized by score).
    Vertical dashed lines = sensor positions (labelled by arrival time).
    Shows each vehicle's journey across the bridge left-to-right.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    n_veh  = len(vehicles)
    cmap   = cm.get_cmap("tab10", n_veh)
    colors = {v.vehicle_id: cmap(i) for i, v in enumerate(vehicles)}

    fig, ax = plt.subplots(figsize=(18, max(5, n_veh * 0.9)))
    fig.suptitle("Suggestion 2 — Vehicle Swimlane Timeline",
                 fontsize=13, fontweight="bold")

    # Draw one lane per vehicle
    for vi, vehicle in enumerate(vehicles):
        lane  = vi
        color = colors[vehicle.vehicle_id]

        # Lane baseline
        ax.axhline(lane, color=color, linewidth=0.6, alpha=0.3)

        # Lane label
        ax.text(
            -0.01, lane,
            f"V{vehicle.vehicle_id}  {vehicle.v_est*3.6:.0f}km/h\n"
            f"{_to_ts(vehicle.entry_ts)}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7, color=color
        )

        # Plot events belonging to this vehicle across all sensors
        for sensor_id, data in sensor_data.items():
            timeline = data["timeline"]
            scores   = data["scores"]

            for ei, event in enumerate(timeline):
                score = scores[ei][vi]
                if score < 0.05:
                    continue
                size = 40 + score * 200   # bigger dot = higher score
                ax.scatter(
                    event.timestamp, lane,
                    s=size, color=color, alpha=0.85,
                    edgecolors="black", linewidths=0.5, zorder=4
                )
                ax.text(
                    event.timestamp, lane + 0.08,
                    f"{score:.2f}",
                    ha="center", fontsize=6, color=color
                )

    # Draw vertical dashed lines for each sensor (at mu for V0 as reference)
    for sensor_id, data in sensor_data.items():
        _, t_min, t_max, mu, sigma = data["mus"][0]
        ax.axvline(mu, color="grey", linestyle="--",
                   linewidth=0.7, alpha=0.4)
        ax.text(mu, n_veh - 0.1,
                f"{sensor_id[:8]}\n{data['pos']:.0f}m",
                ha="center", fontsize=6, color="grey",
                rotation=0)

    ax.set_yticks(range(n_veh))
    ax.set_yticklabels([f"V{v.vehicle_id}" for v in vehicles], fontsize=8)
    ax.set_xlabel("Timestamp", fontsize=9)
    ax.set_ylabel("Vehicle lane", fontsize=9)
    ax.set_ylim(-0.6, n_veh - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    raw_ticks = ax.get_xticks()
    ax.set_xticklabels(
        [_to_ts(t) for t in raw_ticks],
        rotation=30, ha="right", fontsize=7
    )

    plt.tight_layout()
    plt.savefig("viz_options/viz2_swimlane.png", dpi=150, bbox_inches="tight")
    print("[VIZ2] Saved: viz2_swimlane.png")
    plt.show()


# ---------------------------------------------------------------------------
# Suggestion 3 — Per-sensor Gaussian Probability Curves
# ---------------------------------------------------------------------------

def viz_gaussian_curves(sensor_data: dict, vehicles: list[Vehicle]) -> None:
    """
    One subplot per sensor, stacked vertically.
    X = timestamp.
    Y = Gaussian probability score (0 to 1).
    One coloured curve per vehicle showing expected arrival distribution.
    Vertical markers where real events land — height = score at that point.
    Peaks = expected arrival time. Events at peak = perfect match.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    sensors = list(sensor_data.keys())
    n_sens  = len(sensors)
    n_veh   = len(vehicles)
    if n_sens == 0:
        print("[VIZ3] No data"); return

    cmap   = cm.get_cmap("tab10", n_veh)
    colors = {v.vehicle_id: cmap(i) for i, v in enumerate(vehicles)}

    fig, axes = plt.subplots(
        n_sens, 1,
        figsize=(16, n_sens * 3.5),
        sharex=False, squeeze=False
    )
    fig.suptitle("Gaussian Arrival Curves per Sensor",
                 fontsize=13, fontweight="bold")

    for si, sensor_id in enumerate(sensors):
        data     = sensor_data[sensor_id]
        timeline = data["timeline"]
        mus      = data["mus"]
        scores   = data["scores"]
        ax       = axes[si][0]

        # Time range: span all vehicle windows + event times
        all_times = [e.timestamp for e in timeline]
        for _, t_min, t_max, mu, sigma in mus:
            all_times += [t_min - sigma, t_max + sigma]

        t_lo = min(all_times)
        t_hi = max(all_times)
        t_range = np.linspace(t_lo, t_hi, 500)

        # Draw Gaussian curve for each vehicle
        for vehicle in vehicles:
            vid, t_min, t_max, mu, sigma = mus[vehicle.vehicle_id]
            color  = colors[vid]
            curve  = np.exp(-0.5 * ((t_range - mu) / sigma) ** 2)
            ax.plot(t_range, curve, color=color,
                    linewidth=1.8, alpha=0.85,
                    label=f"V{vid}  mu={_to_ts(mu)}  σ={sigma:.1f}s")
            ax.fill_between(t_range, curve, alpha=0.06, color=color)
            # mu marker
            ax.axvline(mu, color=color, linewidth=0.8,
                       linestyle=":", alpha=0.6)

        # Plot real events as vertical stems
        for ei, event in enumerate(timeline):
            best_vi = int(max(range(n_veh), key=lambda i: scores[ei][i]))
            best_sc = scores[ei][best_vi]
            color   = colors[vehicles[best_vi].vehicle_id]
            ax.vlines(event.timestamp, 0, best_sc,
                      color=color, linewidth=2, zorder=5)
            ax.scatter(event.timestamp, best_sc,
                       color=color, s=50, zorder=6,
                       edgecolors="black", linewidths=0.5)
            ax.text(event.timestamp, best_sc + 0.03,
                    f"{_to_ts(event.timestamp)}\n{best_sc:.2f}",
                    ha="center", fontsize=6, color=color)

        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("Probability score", fontsize=8)
        ax.set_title(
            f"Sensor {sensor_id}  pos={data['pos']:.1f}m",
            fontsize=9
        )
        ax.legend(fontsize=6, loc="upper right",
                  framealpha=0.8, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        raw_ticks = ax.get_xticks()
        ax.set_xticklabels(
            [_to_ts(t) for t in raw_ticks],
            rotation=25, ha="right", fontsize=7
        )

    plt.tight_layout()
    plt.savefig("viz_options/viz3_gaussian_curves.png", dpi=150, bbox_inches="tight")
    print("[VIZ3] Saved: viz3_gaussian_curves.png")
    plt.show()


# ---------------------------------------------------------------------------
# Suggestion 4 — Event Ownership Table
# ---------------------------------------------------------------------------

def viz_ownership_table(sensor_data: dict, vehicles: list[Vehicle]) -> None:
    """
    One table per sensor.
    Rows = events (timestamp, amplitude).
    Columns = vehicles.
    Cell = probability score with background colour intensity.
    Highest score in each row outlined in bold — the "owner" vehicle.
    Clean, no clutter, works at any scale.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    sensors = list(sensor_data.keys())
    n_sens  = len(sensors)
    n_veh   = len(vehicles)
    if n_sens == 0:
        print("[VIZ4] No data"); return

    cmap    = cm.get_cmap("tab10", n_veh)
    v_colors = [cmap(i) for i in range(n_veh)]

    # One figure, one subplot per sensor stacked vertically
    row_h   = 0.45
    fig_h   = sum(
        max(2.0, len(sensor_data[s]["timeline"]) * row_h + 1.5)
        for s in sensors
    )
    fig, axes = plt.subplots(
        n_sens, 1,
        figsize=(max(10, n_veh * 1.8 + 4), fig_h),
        squeeze=False
    )
    fig.suptitle("Events probabilities Table",
                 fontsize=13, fontweight="bold")

    score_cmap = plt.cm.YlOrRd

    for si, sensor_id in enumerate(sensors):
        data     = sensor_data[sensor_id]
        timeline = data["timeline"]
        scores   = data["scores"]
        ax       = axes[si][0]
        ax.axis("off")

        n_events = len(timeline)
        if n_events == 0:
            ax.set_title(f"Sensor {sensor_id} — no events", fontsize=9)
            continue

        # Column headers: row labels + one per vehicle
        col_labels = ["Timestamp", "Amplitude"] + \
                     [f"V{v.vehicle_id}\n{v.v_est*3.6:.0f}km/h"
                      for v in vehicles]

        # Build table data
        cell_text   = []
        cell_colors = []

        for ei, event in enumerate(timeline):
            row_scores = scores[ei]
            best_vi    = int(np.argmax(row_scores))

            text_row   = [
                _to_ts(event.timestamp),
                f"{event.peak_amplitude:+.5f}"
            ]
            color_row  = ["#f0f0f0", "#f0f0f0"]

            for vi, sc in enumerate(row_scores):
                text_row.append(f"{sc:.3f}")
                # Background intensity from score
                rgba = list(score_cmap(sc))
                rgba[3] = 0.75
                color_row.append(rgba)

            cell_text.append(text_row)
            cell_colors.append(color_row)

        tbl = ax.table(
            cellText   = cell_text,
            cellColours= cell_colors,
            colLabels  = col_labels,
            cellLoc    = "center",
            loc        = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.6)

        # Bold border on highest-score vehicle cell per row
        for ei in range(n_events):
            best_vi = int(np.argmax(scores[ei]))
            cell    = tbl[(ei + 1, best_vi + 2)]  # +1 header, +2 label cols
            cell.set_edgecolor("black")
            cell.set_linewidth(2.0)

        # Header row styling
        for ci in range(len(col_labels)):
            header = tbl[(0, ci)]
            header.set_facecolor("#2c3e50")
            header.set_text_props(color="white", fontweight="bold")

        ax.set_title(
            f"Sensor {sensor_id}  pos={data['pos']:.1f}m  "
            f"({n_events} events)",
            fontsize=9, pad=8, loc="left"
        )

    plt.tight_layout()
    plt.savefig("viz_options/viz4_probabilities_table.png", dpi=150, bbox_inches="tight")
    print("[VIZ4] Saved: viz4_probabilities_table.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sensor_positions = load_sensor_positions()
    vehicles         = load_wim_entries()
    sensor_events    = load_sensor_events()

    vehicles = run_tracker(sensor_positions, sensor_events, vehicles)
    # summarise(vehicles)

    event_timeline_with_probabilities(sensor_positions, sensor_events, vehicles)

    # Build shared score data once — used by all visualisations
    sensor_data = _build_sensor_data(sensor_positions, sensor_events, vehicles)

    viz_heatmap(sensor_data, vehicles)
    viz_swimlane(sensor_data, sensor_positions, vehicles)
    viz_gaussian_curves(sensor_data, vehicles)
    viz_ownership_table(sensor_data, vehicles)