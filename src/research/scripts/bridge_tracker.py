"""
bridge_tracker.py  —  PROBABILISTIC VEHICLE TRACKING
=====================================================
Scores sensor events against vehicle tracks using three methods:

    _build_sensor_data          — WIM Gaussian only (stateless baseline)
    _build_sensor_data_propagated — Bayesian propagation with velocity refinement
    _build_sensor_data_dual     — Score A (WIM) + Score B (hop-derived) + Score Bayes

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

from src.shared import config
from src.shared.bridge_model import *

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

VELOCITY_TOLERANCE_KMH = 40.0    # ±km/h applied to WIM speed → v_min / v_max
TIMESTAMP_FORMAT       = "%Y-%m-%d %H:%M:%S"
_SENSOR_COL_SUFFIX     = "_dominant_peaks"

WIM_GATE_POSITION_M    = -2300.0  # Physical position of WIM gate in bridge coords (m)
                                   # Derived: v1 actual = (238.9-99.8)/(154.56-148.87)
                                   #          = 24.45 m/s → gate = 99.8 - 24.45×148.87

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
    sensor_id  : str
    sensor_pos : float
    t_min      : float
    t_max      : float
    mu         : float
    matched    : bool
    event_ts   : Optional[float] = None
    event_amp  : Optional[float] = None
    time_error : Optional[float] = None


@dataclass
class Vehicle:
    vehicle_id : int
    entry_ts   : float
    v_min      : float
    v_max      : float
    mass_kg    : float
    v_est      : float
    anchor_pos : float
    anchor_ts  : float
    log        : list = field(default_factory=list)
    status     : str  = "ACTIVE"

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_reference_epoch: Optional[datetime] = None


def _dt_to_seconds(dt: datetime) -> float:
    global _reference_epoch
    if _reference_epoch is None:
        _reference_epoch = dt
    return (dt - _reference_epoch).total_seconds()


def _to_ts(seconds: float) -> str:
    if _reference_epoch is None:
        return f"{seconds:.3f}s"
    dt = _reference_epoch + timedelta(seconds=seconds)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _parse_str_to_dt(ts_str: str) -> datetime:
    for fmt in (
        TIMESTAMP_FORMAT + ".%f",
        TIMESTAMP_FORMAT,
        "%Y-%d-%m %H:%M:%S.%f",
        "%Y-%d-%m %H:%M:%S",
        "%d/%m/%Y %H:%M:%S"

    ):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: '{ts_str}'")


def _parse_timestamp(ts_str: str) -> float:
    return _dt_to_seconds(_parse_str_to_dt(ts_str))

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_sensor_positions(limit) -> dict[str, float]:
    """Boundary sensors only — tail + entry at every span junction."""
    bridge    = load_bridge(config.position_csv, config.threshold_csv)
    junctions = bridge.find_boundaries()

    positions = {}
    for junction in junctions[:limit]:
        for sensor in junction.all_sensors():
            positions[sensor.sensor_id] = sensor.distance

    print(f"[LOAD] {len(junctions[:2])} span junctions")
    print(f"[LOAD] {len(positions)} boundary sensors")
    return positions


def load_wim_entries() -> list[Vehicle]:
    """
    MUST be called before load_sensor_events() — sets reference epoch.
    vehicle_id: sequential 0, 1, 2, ... (CSV index ignored)
    """
    vehicles = []
    with open(config.WIM_CSV, newline="", encoding="utf-8-sig") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            if not row.get("Velocity"):
                continue
            speed_kmh = float(row["Velocity"])
            entry_ts  = _parse_timestamp(row["StartTimeStr"])
            vehicles.append(Vehicle(
                vehicle_id = idx,
                entry_ts   = entry_ts,
                v_min      = (speed_kmh - VELOCITY_TOLERANCE_KMH) / 3.6,
                v_max      = (speed_kmh + VELOCITY_TOLERANCE_KMH) / 3.6,
                mass_kg    = float(row["GrossWeight"]),
                v_est      = speed_kmh / 3.6,
                anchor_pos = WIM_GATE_POSITION_M,
                anchor_ts  = entry_ts,
            ))
            print(f"  [WIM] V{idx}: "
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
    events     : dict[str, list[SensorEvent]] = {}
    seen       : dict[str, set[float]]        = {}
    skipped    = 0
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
        print(f"[LOAD] {duplicates} duplicate events removed")
    print(f"[LOAD] {total} unique events across {len(events)} sensors")
    return events

# ---------------------------------------------------------------------------
# Scoring — baseline: always from WIM gate
# ---------------------------------------------------------------------------

def _build_sensor_data(
    sensor_positions : dict[str, float],
    sensor_events    : dict[str, list[SensorEvent]],
    vehicles         : list[Vehicle],
) -> dict:
    """
    Stateless baseline. All windows projected from WIM gate using WIM velocity.
    Returns sensor_data dict keyed by sensor_id:
        pos      : float
        timeline : [SensorEvent]
        scores   : [[float]]      scores[event_idx][vehicle_idx]
        mus      : [(vid, t_min, t_max, mu, sigma)]
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
# Scoring — propagated: Bayesian anchor propagation with velocity refinement
# ---------------------------------------------------------------------------

def _build_sensor_data_propagated(
    sensor_positions : dict[str, float],
    sensor_events    : dict[str, list[SensorEvent]],
    vehicles         : list[Vehicle],
) -> dict:
    """
    Bayesian propagation version. Drop-in replacement for _build_sensor_data.
    Anchor advances after each matched sensor. Velocity refined per match.
    Velocity refinement currently DISABLED for diagnostic purposes.
    """
    V_SIGMA_FLOOR       = 5
    MIN_SCORE_TO_UPDATE = 0.0005
    MIN_SIGMA_PROP = 2.0  

    ordered     = sorted(sensor_positions.items(), key=lambda x: x[1])
    sensor_data = {}

    states = {}
    for vehicle in vehicles:
        v_half = (vehicle.v_max - vehicle.v_min) / 2
        states[vehicle.vehicle_id] = {
            "anchor_pos" : WIM_GATE_POSITION_M,
            "anchor_ts"  : vehicle.entry_ts,
            "v_est"      : vehicle.v_est,
            "v_sigma"    : v_half,
            "prior"      : 1.0,
        }

    for sensor_id, sensor_pos in ordered:
        events = sensor_events.get(sensor_id, [])
        if not events:
            sensor_data[sensor_id] = {
                "pos": sensor_pos, "timeline": [],
                "scores": [], "mus": [],
            }
            continue

        timeline = sorted(events, key=lambda e: e.timestamp)
        mus      = []

        for vehicle in vehicles:
            st    = states[vehicle.vehicle_id]
            d     = sensor_pos - st["anchor_pos"]

            v_min_eff = max(vehicle.v_min, st["v_est"] - st["v_sigma"])
            v_max_eff = min(vehicle.v_max, st["v_est"] + st["v_sigma"])
            if v_min_eff >= v_max_eff:
                v_min_eff = vehicle.v_min
                v_max_eff = vehicle.v_max

            t_min = st["anchor_ts"] + d / v_max_eff
            t_max = st["anchor_ts"] + d / v_min_eff
            t_min, t_max = min(t_min, t_max), max(t_min, t_max)
            mu    = (t_min + t_max) / 2
            sigma = max((t_max - t_min) / 2, MIN_SIGMA_PROP)
            mus.append((vehicle.vehicle_id, t_min, t_max, mu, sigma))

        scores = []
        for event in timeline:
            row = []
            for vi, vehicle in enumerate(vehicles):
                _, t_min, t_max, mu, sigma = mus[vi]
                raw = math.exp(-0.5 * ((event.timestamp - mu) / sigma) ** 2)
                row.append(raw)
            scores.append(row)

        cumulative_confidence = {}
        for vi, vehicle in enumerate(vehicles):
            st       = states[vehicle.vehicle_id]
            _, t_min, t_max, mu, sigma = mus[vi]
            best_raw = max(
                math.exp(-0.5 * ((e.timestamp - mu) / sigma) ** 2)
                for e in timeline
            ) if timeline else 0.0
            cumulative_confidence[vehicle.vehicle_id] = st["prior"] * best_raw

        sensor_data[sensor_id] = {
            "pos"                   : sensor_pos,
            "timeline"              : timeline,
            "scores"                : scores,
            "mus"                   : mus,
            "cumulative_confidence" : cumulative_confidence,
        }

        for vi, vehicle in enumerate(vehicles):
            st  = states[vehicle.vehicle_id]
            _, t_min, t_max, mu, sigma = mus[vi]
            if not timeline:
                continue

            raw_scores = [
                math.exp(-0.5 * ((e.timestamp - mu) / sigma) ** 2)
                for e in timeline
            ]
            best_ei    = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
            best_score = raw_scores[best_ei]
            best_event = timeline[best_ei]

            if best_score < MIN_SCORE_TO_UPDATE:
                continue

            # Velocity refinement — DISABLED for diagnostic
            # # Uncomment to re-enable:
            # d       = sensor_pos - st["anchor_pos"]
            # delta_t = best_event.timestamp - st["anchor_ts"]
            # if delta_t > 0:
            #     v_measured  = d / delta_t
            #     v_measured  = max(vehicle.v_min, min(vehicle.v_max, v_measured))
            #     v_refined   = best_score * v_measured + (1-best_score) * st["v_est"]
            #     v_refined   = max(vehicle.v_min, min(vehicle.v_max, v_refined))
            #     v_sigma_new = st["v_sigma"] * (1 - best_score)
            #     v_sigma_new = max(V_SIGMA_FLOOR, v_sigma_new)
            #     st["v_est"]   = v_refined
            #     st["v_sigma"] = v_sigma_new

            st["anchor_pos"] = sensor_pos
            st["anchor_ts"]  = best_event.timestamp
            prior_before     = st["prior"]
            posterior        = prior_before * best_score
            st["prior"]      = best_score

            print(f"  [PROP] V{vehicle.vehicle_id}  "
                  f"sensor={sensor_id}  "
                  f"event={_to_ts(best_event.timestamp)}  "
                  f"prior={prior_before:.4f}  "
                  f"likelihood={best_score:.4f}  "
                  f"posterior={posterior:.4f}  "
                  f"v_est={st['v_est']*3.6:.1f}km/h  "
                  f"v_sigma={st['v_sigma']:.3f}m/s")

    return sensor_data


# ---------------------------------------------------------------------------
# Scoring — dual: Score A (WIM) + Score B (hop-derived) + Score Bayes
# ---------------------------------------------------------------------------

def _build_sensor_data_dual(
    sensor_positions : dict[str, float],
    sensor_events    : dict[str, list[SensorEvent]],
    vehicles         : list[Vehicle],
) -> dict:
    """
    Computes three scores per (sensor, event, vehicle):

    Score A  —  WIM Gaussian  (scores_wim)
        Projected from WIM gate using WIM-recorded velocity ± 10 km/h.
        Stateless — always available.

    Score B  —  Propagated Gaussian  (scores_prop)
        Projected from last anchor using hop-derived velocity ± 10 km/h.
        None at sensor 1. Initialised from best Score A match at sensor 1.
        SCORE_B_MARGIN_S added each side to widen window.

    Score Bayes  —  Bayesian posterior  (scores_bayes)
        posterior = prior × Score B
        Falls back to Score A at sensor 1.
        Prior carried forward unchanged on skip (below threshold).

    Fixes:
        Fix 1 — prev_anchor only advances when sensor position changes
        Fix 2 — anchor only advances if event timestamp is strictly later

    Output per sensor_id:
        pos          : float
        timeline     : [SensorEvent]
        scores_wim   : [[float]]
        scores_prop  : [[float | None]]
        scores_bayes : [[float]]
        mus_wim      : [(vid, t_min, t_max, mu, sigma)]
        mus_prop     : [(vid, t_min, t_max, mu, sigma) | None]
    """

    MIN_SCORE_TO_UPDATE = 0.0005
    VELOCITY_TOL_MS     = VELOCITY_TOLERANCE_KMH / 3.6
    MIN_SIGMA_PROP      = 2.5    # minimum Score B sigma (co-located sensors)
    SCORE_B_MARGIN_S    = 2.5    # seconds added each side of Score B window

    ordered     = sorted(sensor_positions.items(), key=lambda x: x[1])
    sensor_data = {}

    states = {}
    for vehicle in vehicles:
        states[vehicle.vehicle_id] = {
            "prev_anchor" : {"pos": WIM_GATE_POSITION_M, "ts": vehicle.entry_ts},
            "last_anchor" : None,
            "prior"       : 1.0,
        }

    for si, (sensor_id, sensor_pos) in enumerate(ordered):

        events = sensor_events.get(sensor_id, [])
        if not events:
            sensor_data[sensor_id] = {
                "pos"          : sensor_pos,
                "timeline"     : [],
                "scores_wim"   : [],
                "scores_prop"  : [],
                "scores_bayes" : [],
                "mus_wim"      : [],
                "mus_prop"     : [],
            }
            continue

        timeline   = sorted(events, key=lambda e: e.timestamp)
        d_from_wim = sensor_pos - WIM_GATE_POSITION_M

        mus_wim  = []
        mus_prop = []

        for vehicle in vehicles:
            st = states[vehicle.vehicle_id]

            # Score A window — WIM gate, WIM velocity
            t_min_w = vehicle.entry_ts + d_from_wim / vehicle.v_max
            t_max_w = vehicle.entry_ts + d_from_wim / vehicle.v_min
            t_min_w, t_max_w = min(t_min_w, t_max_w), max(t_min_w, t_max_w)
            mu_w    = (t_min_w + t_max_w) / 2
            sigma_w = (t_max_w - t_min_w) / 2 or 1e-3
            mus_wim.append((vehicle.vehicle_id, t_min_w, t_max_w, mu_w, sigma_w))

            # Score B window — last anchor, hop-derived velocity
            if st["last_anchor"] is None:
                if si > 0:
                    print(f"  [DEBUG-NONE] V{vehicle.vehicle_id}  "
                          f"sensor={sensor_id}  → last_anchor is None")
                mus_prop.append(None)
                continue

            prev   = st["prev_anchor"]
            last   = st["last_anchor"]
            d_hop  = last["pos"] - prev["pos"]
            dt_hop = last["ts"]  - prev["ts"]

            if dt_hop <= 0:
                print(f"  [DEBUG-NONE] V{vehicle.vehicle_id}  "
                      f"sensor={sensor_id}  "
                      f"→ dt_hop={dt_hop:.4f}s <= 0  (clock sync issue)")
                mus_prop.append(None)
                continue

            if d_hop <= 0:
                print(f"  [DEBUG-NONE] V{vehicle.vehicle_id}  "
                      f"sensor={sensor_id}  "
                      f"→ d_hop={d_hop:.2f}m <= 0")
                mus_prop.append(None)
                continue

            v_derived = d_hop / dt_hop
            v_min_p   = max(v_derived - VELOCITY_TOL_MS, 1e-3)
            v_max_p   = v_derived + VELOCITY_TOL_MS

            d_prop  = sensor_pos - last["pos"]
            t_min_p = last["ts"] + d_prop / v_max_p
            t_max_p = last["ts"] + d_prop / v_min_p
            t_min_p, t_max_p = min(t_min_p, t_max_p), max(t_min_p, t_max_p)

            # Add margin — widens window without shifting mu
            t_min_p -= SCORE_B_MARGIN_S
            t_max_p += SCORE_B_MARGIN_S

            mu_p    = (t_min_p + t_max_p) / 2
            sigma_p = max((t_max_p - t_min_p) / 2, MIN_SIGMA_PROP)
            mus_prop.append((vehicle.vehicle_id, t_min_p, t_max_p, mu_p, sigma_p))

            # Window comparison diagnostic
            if si >= 1:
                mu_shift = mu_p - mu_w
                overlap  = max(0.0,
                    min(mu_w + sigma_w, mu_p + sigma_p) -
                    max(mu_w - sigma_w, mu_p - sigma_p)
                )
                print(f"  [DEBUG-WIN]  V{vehicle.vehicle_id}  "
                      f"sensor={sensor_id}  si={si}  "
                      f"d_hop={d_hop:.1f}m  dt_hop={dt_hop:.3f}s  "
                      f"v_derived={v_derived*3.6:.1f}km/h  d_prop={d_prop:.1f}m  "
                      f"| mu_A={_to_ts(mu_w)}  mu_B={_to_ts(mu_p)}  "
                      f"mu_shift={mu_shift:+.3f}s  "
                      f"| sigma_A={sigma_w:.3f}s  sigma_B={sigma_p:.3f}s  "
                      f"| overlap={overlap:.3f}s")

        # Events vs Score B windows diagnostic
        if si >= 1:
            print(f"  [DEBUG-EVENTS] sensor={sensor_id}  {len(timeline)} events:")
            for event in timeline:
                print(f"    event={_to_ts(event.timestamp)}  "
                      f"amp={event.peak_amplitude:+.5f}")
                for vi, vehicle in enumerate(vehicles):
                    if mus_prop[vi] is None:
                        print(f"      V{vehicle.vehicle_id}  Score B window=None")
                    else:
                        _, t_min_p, t_max_p, mu_p, sigma_p = mus_prop[vi]
                        inside = t_min_p <= event.timestamp <= t_max_p
                        dist   = event.timestamp - mu_p
                        sc_b   = math.exp(
                            -0.5 * ((event.timestamp - mu_p) / sigma_p) ** 2
                        )
                        print(f"      V{vehicle.vehicle_id}  "
                              f"window=[{_to_ts(t_min_p)}, {_to_ts(t_max_p)}]  "
                              f"inside={inside}  "
                              f"dist_from_mu={dist:+.3f}s  "
                              f"score_B={sc_b:.4f}")

        # Scores per event
        scores_wim   = []
        scores_prop  = []
        scores_bayes = []

        for event in timeline:
            row_wim   = []
            row_prop  = []
            row_bayes = []

            for vi, vehicle in enumerate(vehicles):
                st = states[vehicle.vehicle_id]

                _, t_min_w, t_max_w, mu_w, sigma_w = mus_wim[vi]
                sc_a = math.exp(
                    -0.5 * ((event.timestamp - mu_w) / sigma_w) ** 2
                )
                row_wim.append(sc_a)

                if mus_prop[vi] is None:
                    sc_b = None
                else:
                    _, t_min_p, t_max_p, mu_p, sigma_p = mus_prop[vi]
                    sc_b = math.exp(
                        -0.5 * ((event.timestamp - mu_p) / sigma_p) ** 2
                    )
                row_prop.append(sc_b)

                sc_bayes = sc_a if sc_b is None else st["prior"] * sc_b
                row_bayes.append(sc_bayes)

            scores_wim.append(row_wim)
            scores_prop.append(row_prop)
            scores_bayes.append(row_bayes)

        sensor_data[sensor_id] = {
            "pos"          : sensor_pos,
            "timeline"     : timeline,
            "scores_wim"   : scores_wim,
            "scores_prop"  : scores_prop,
            "scores_bayes" : scores_bayes,
            "mus_wim"      : mus_wim,
            "mus_prop"     : mus_prop,
        }

        # State update per vehicle
        for vi, vehicle in enumerate(vehicles):
            st = states[vehicle.vehicle_id]

            if si == 0:
                # Sensor 1: initialise Score B chain from best Score A
                sc_a_vals = [scores_wim[ei][vi] for ei in range(len(timeline))]
                best_sc_a = max(sc_a_vals)
                if best_sc_a < MIN_SCORE_TO_UPDATE:
                    print(f"  [DEBUG-NONE] V{vehicle.vehicle_id}  "
                          f"sensor={sensor_id}  "
                          f"best_score_A={best_sc_a:.4f} < {MIN_SCORE_TO_UPDATE}  "
                          f"→ Score B chain NOT initialised")
                    continue
                best_ei    = int(sc_a_vals.index(best_sc_a))
                best_event = timeline[best_ei]
                st["last_anchor"] = {"pos": sensor_pos, "ts": best_event.timestamp}
                print(f"  [BAYES-INIT] V{vehicle.vehicle_id}  "
                      f"sensor={sensor_id}  "
                      f"event={_to_ts(best_event.timestamp)}  "
                      f"score_A={best_sc_a:.4f}  "
                      f"→ hop chain initialised  prior=1.0")

            else:
                # Sensor 2+: Bayesian update from best Score B match
                if mus_prop[vi] is None:
                    continue

                sc_b_vals  = [
                    scores_prop[ei][vi] if scores_prop[ei][vi] is not None else 0.0
                    for ei in range(len(timeline))
                ]
                best_sc_b  = max(sc_b_vals)
                best_ei    = int(sc_b_vals.index(best_sc_b))
                best_event = timeline[best_ei]
                prior_before = st["prior"]
                posterior    = prior_before * best_sc_b

                if best_sc_b < MIN_SCORE_TO_UPDATE:
                    print(f"  [BAYES-SKIP] V{vehicle.vehicle_id}  "
                          f"sensor={sensor_id}  "
                          f"likelihood={best_sc_b:.4f} < {MIN_SCORE_TO_UPDATE}  "
                          f"prior carried forward={prior_before:.4f}  "
                          f"→ anchor held at {st['last_anchor']['pos']:.1f}m")
                    continue

                # Fix 2 — backwards timestamp guard
                if best_event.timestamp <= st["last_anchor"]["ts"]:
                    print(f"  [FIX2-SKIP]  V{vehicle.vehicle_id}  "
                          f"sensor={sensor_id}  "
                          f"event={_to_ts(best_event.timestamp)}  "
                          f"<= last_anchor.ts={_to_ts(st['last_anchor']['ts'])}  "
                          f"→ anchor NOT updated (backwards timestamp)")
                    continue

                d_hop_log  = sensor_pos - st["last_anchor"]["pos"]
                dt_hop_log = best_event.timestamp - st["last_anchor"]["ts"]
                v_log_kmh  = (
                    d_hop_log / dt_hop_log * 3.6
                    if dt_hop_log > 0 else float("nan")
                )

                # Fix 1 — only advance prev_anchor on position change
                if sensor_pos != st["last_anchor"]["pos"]:
                    st["prev_anchor"] = st["last_anchor"]
                    fix1_note = ""
                else:
                    fix1_note = "  [FIX1: prev_anchor held — co-located]"

                st["last_anchor"] = {"pos": sensor_pos, "ts": best_event.timestamp}
                st["prior"]       = posterior

                print(f"  [BAYES-PROP] V{vehicle.vehicle_id}  "
                      f"sensor={sensor_id}  "
                      f"event={_to_ts(best_event.timestamp)}  "
                      f"prior={prior_before:.4f}  "
                      f"likelihood={best_sc_b:.4f}  "
                      f"posterior={posterior:.4f}  "
                      f"v_hop={v_log_kmh:.1f} km/h  "
                      f"d_prop={d_hop_log:.1f}m  "
                      f"anchor={sensor_pos:.1f}m"
                      f"{fix1_note}")

    return sensor_data


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
# All three viz functions are compatible with all three scoring functions:
#
#   _build_sensor_data          → keys: scores, mus
#   _build_sensor_data_propagated → keys: scores, mus, cumulative_confidence
#   _build_sensor_data_dual     → keys: scores_wim, mus_wim, scores_prop,
#                                        mus_prop, scores_bayes
#
# Each function calls _normalise_sensor_data() to unify keys before rendering.
# Bayes panel / column / markers are shown only when Bayes data is available.
# ---------------------------------------------------------------------------



def _normalise_sensor_data(data: dict, vehicles: list) -> dict:
    """
    Unify sensor_data keys across all three scoring functions.
 
    Returns a dict with consistent keys:
        scores_wim   : [[float]]       — always present
        mus_wim      : [tuple]         — always present
        scores_bayes : [[float]] | None — None if not available
        mus_prop     : [tuple|None]    — None entries where Score B unavailable
        has_bayes    : bool            — whether to show Bayes panel/column
    """
    # Score A — works for all three functions
    scores_wim = data.get("scores_wim", data.get("scores", []))
    mus_wim    = data.get("mus_wim",    data.get("mus",    []))
 
    # Score B windows — only from _build_sensor_data_dual
    mus_prop = data.get("mus_prop", [])
 
    # Score Bayes — three possible sources:
    scores_bayes = data.get("scores_bayes", None)
 
    # Events close to mu get high Bayes; events far from mu decay to zero.
    if scores_bayes is None and "cumulative_confidence" in data:
        cum      = data["cumulative_confidence"]
        scores   = data.get("scores", [])
        timeline = data.get("timeline", [])
        n_events = len(timeline)
 
        # Best raw score per vehicle across all events at this sensor
        best_raw_per_vehicle = {}
        for vi, v in enumerate(vehicles):
            best = max(
                (scores[ei][vi] for ei in range(n_events)),
                default=0.0
            ) if scores else 0.0
            best_raw_per_vehicle[v.vehicle_id] = best
 
        # Per-event Bayes score = prior x raw_gaussian
        scores_bayes = []
        for ei in range(n_events):
            row = []
            for vi, v in enumerate(vehicles):
                raw_sc   = scores[ei][vi] if scores else 0.0
                best_raw = best_raw_per_vehicle[v.vehicle_id]
                cum_conf = cum.get(v.vehicle_id, 0.0)
                prior    = (cum_conf / best_raw) if best_raw > 0 else 0.0
                row.append(prior * raw_sc)
            scores_bayes.append(row)
 
    has_bayes = (
        scores_bayes is not None
        and len(scores_bayes) > 0
    )
 
    return {
        "scores_wim"  : scores_wim,
        "mus_wim"     : mus_wim,
        "scores_bayes": scores_bayes,
        "mus_prop"    : mus_prop,
        "has_bayes"   : has_bayes,
    }
 

def viz_heatmap(sensor_data: dict, vehicles: list,
                output_dir: str = ".") -> None:
    """
    Per sensor:
      Always : Score A heatmap (WIM Gaussian)
      If Bayes available : second heatmap for Score Bayes

    Compatible with all three scoring functions.
    Saved: <output_dir>/viz1_heatmap.png
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    sensors = list(sensor_data.keys())
    n_sens  = len(sensors)
    if n_sens == 0:
        print("[VIZ1] No data"); return

    # Check if any sensor has Bayes data
    has_bayes_any = any(
        _normalise_sensor_data(sensor_data[s], vehicles)["has_bayes"]
        for s in sensors
    )
    n_cols_fig = 2 if has_bayes_any else 1
    n_veh      = len(vehicles)
    n_events   = max((len(sensor_data[s]["timeline"]) for s in sensors), default=1)

    fig, axes = plt.subplots(
        n_sens, n_cols_fig,
        figsize=(max(10, n_events * 1.8) * n_cols_fig,
                 n_sens * (n_veh * 0.6 + 1.5)),
        squeeze=False,
        gridspec_kw={"wspace": 0.35}
    )

    title = "Heatmap — Score A (WIM Gaussian)"
    if has_bayes_any:
        title += "  |  Score Bayes (Posterior)"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    score_cmap = plt.cm.YlOrRd
    y_labels   = [
        f"V{v.vehicle_id}  {v.v_est*3.6:.0f}km/h  {_to_ts(v.entry_ts)}"
        for v in vehicles
    ]

    for si, sensor_id in enumerate(sensors):
        data       = sensor_data[sensor_id]
        timeline   = data["timeline"]
        n_ev       = len(timeline)
        sensor_pos = data["pos"]
        x_labels   = [_to_ts(e.timestamp) for e in timeline]
        nd         = _normalise_sensor_data(data, vehicles)

        mat_a = np.zeros((n_veh, max(n_ev, 1)))
        mat_y = np.zeros((n_veh, max(n_ev, 1)))
        for ei in range(n_ev):
            for vi in range(n_veh):
                mat_a[vi, ei] = nd["scores_wim"][ei][vi] if nd["scores_wim"] else 0.0
                if nd["has_bayes"]:
                    mat_y[vi, ei] = nd["scores_bayes"][ei][vi]

        def _draw(ax, matrix, title_suffix):
            im = ax.imshow(matrix, aspect="auto", cmap=score_cmap,
                           vmin=0, vmax=1, interpolation="nearest")
            for vi in range(n_veh):
                for ci in range(n_ev):
                    val = matrix[vi, ci]
                    ax.text(ci, vi, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if val > 0.6 else "black")
            ax.set_yticks(range(n_veh))
            ax.set_yticklabels(y_labels, fontsize=7)
            ax.set_xticks(range(n_ev))
            ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"Sensor {sensor_id}  {sensor_pos:.1f}m  |  {title_suffix}",
                         fontsize=9, pad=4)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("Score", fontsize=7)

        _draw(axes[si][0], mat_a, "Score A — WIM Gaussian")
        if has_bayes_any:
            _draw(axes[si][1], mat_y,
                  "Score Bayes — Posterior" if nd["has_bayes"] else "Score Bayes — N/A")

    #plt.tight_layout()
    out1 = os.path.join(output_dir, "viz1_heatmap.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"[VIZ1] Saved: {out1}")
    plt.show()


def viz_gaussian_curves(sensor_data: dict, vehicles: list,
                        output_dir: str = ".") -> None:
    """
    One subplot per sensor.
    Solid curve   = Score A Gaussian  (always)
    Dashed curve  = Score B Gaussian  (dual only, where available)
    Filled circle = best Score A vehicle per event  (always)
    Hollow diamond = best Score Bayes vehicle per event  (when available)
 
    Compatible with all three scoring functions.
    Saved: <output_dir>/viz3_gaussian_curves.png
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
 
    sensors = list(sensor_data.keys())
    n_sens  = len(sensors)
    n_veh   = len(vehicles)
    if n_sens == 0:
        print("[VIZ3] No data"); return
 
    cmap   = cm.get_cmap("tab10", n_veh)
    colors = {v.vehicle_id: cmap(i) for i, v in enumerate(vehicles)}
 
    # ── Compute GLOBAL time range across ALL sensors ──────────────────────
    # Used to set the same x-axis on every subplot for easy cross-sensor
    # comparison of event timing.
    global_times = []
    for sensor_id in sensors:
        data     = sensor_data[sensor_id]
        nd       = _normalise_sensor_data(data, vehicles)
        mus_wim  = nd["mus_wim"]
        mus_prop = nd["mus_prop"]
        global_times += [e.timestamp for e in data["timeline"]]
        for tup in mus_wim:
            if tup:
                _, t_min, t_max, mu, sigma = tup
                global_times += [t_min - sigma, t_max + sigma]
        for tup in mus_prop:
            if tup:
                _, t_min, t_max, mu, sigma = tup
                global_times += [t_min - sigma, t_max + sigma]
 
    if not global_times:
        print("[VIZ3] No time data found"); return
 
    global_t_lo = min(global_times)
    global_t_hi = max(global_times)
    global_t_range = np.linspace(global_t_lo, global_t_hi, 1200)
 
    fig, axes = plt.subplots(n_sens, 1, figsize=(16, n_sens * 4.0),
                             sharex=True, squeeze=False)
    fig.suptitle(
        "Gaussian Curves — Score A (solid)  Score B (dashed, dual only)"
        "  Score Bayes on events (◇, when available)",
        fontsize=13, fontweight="bold"
    )
 
    for si, sensor_id in enumerate(sensors):
        data     = sensor_data[sensor_id]
        timeline = data["timeline"]
        nd       = _normalise_sensor_data(data, vehicles)
        mus_wim  = nd["mus_wim"]
        mus_prop = nd["mus_prop"]
        ax       = axes[si][0]
 
        if not timeline:
            ax.set_title(f"Sensor {sensor_id}  {data['pos']:.1f}m  — no events",
                         fontsize=9)
            continue
 
        t_range        = global_t_range
        legend_handles = []
 
        for vi, vehicle in enumerate(vehicles):
            color = colors[vehicle.vehicle_id]
 
            # Score A — solid curve (always)
            if mus_wim and mus_wim[vi] is not None:
                _, t_min_w, t_max_w, mu_w, sigma_w = mus_wim[vi]
                curve_a = np.exp(-0.5 * ((t_range - mu_w) / sigma_w) ** 2)
                line_a, = ax.plot(t_range, curve_a, color=color,
                                  linewidth=1.8, alpha=0.9, linestyle="-",
                                  label=f"V{vehicle.vehicle_id} A  mu={_to_ts(mu_w)}  σ={sigma_w:.1f}s")
                ax.fill_between(t_range, curve_a, alpha=0.05, color=color)
                ax.axvline(mu_w, color=color, linewidth=0.7, linestyle=":", alpha=0.5)
                legend_handles.append(line_a)
 
            # Score B — dashed curve (dual only, where window available)
            if mus_prop and vi < len(mus_prop) and mus_prop[vi] is not None:
                _, t_min_p, t_max_p, mu_p, sigma_p = mus_prop[vi]
                curve_b = np.exp(-0.5 * ((t_range - mu_p) / sigma_p) ** 2)
                line_b, = ax.plot(t_range, curve_b, color=color,
                                  linewidth=1.4, alpha=0.7, linestyle="--",
                                  label=f"V{vehicle.vehicle_id} B  mu={_to_ts(mu_p)}  σ={sigma_p:.1f}s")
                ax.axvline(mu_p, color=color, linewidth=0.7, linestyle="-.", alpha=0.4)
                legend_handles.append(line_b)
 
        for ei, event in enumerate(timeline):
            sc_a_row   = nd["scores_wim"][ei]   if nd["scores_wim"]   else []
            sc_bay_row = nd["scores_bayes"][ei]  if nd["has_bayes"]    else []
 
            # Best Score A — filled circle + stem (always)
            if sc_a_row:
                best_vi_a = int(np.argmax(sc_a_row))
                best_sc_a = sc_a_row[best_vi_a]
                color_a   = colors[vehicles[best_vi_a].vehicle_id]
                ax.vlines(event.timestamp, 0, best_sc_a,
                          color=color_a, linewidth=2.0, zorder=5)
                ax.scatter(event.timestamp, best_sc_a, color=color_a, s=55,
                           zorder=6, edgecolors="black", linewidths=0.5, marker="o")
                ax.text(event.timestamp, best_sc_a + 0.04,
                        f"{_to_ts(event.timestamp)}\nA:{best_sc_a:.2f}",
                        ha="center", fontsize=6, color=color_a)
 
            # Best Score Bayes — hollow diamond (when available)
            if sc_bay_row:
                best_vi_b = int(np.argmax(sc_bay_row))
                best_sc_b = sc_bay_row[best_vi_b]
                color_b   = colors[vehicles[best_vi_b].vehicle_id]
                ax.scatter(event.timestamp, best_sc_b, color="none", s=85,
                           zorder=7, edgecolors=color_b, linewidths=1.8, marker="D")
                ax.text(event.timestamp, best_sc_b - 0.10,
                        f"Bay:{best_sc_b:.2f}",
                        ha="center", fontsize=6, color=color_b, style="italic")
 
        subtitle_parts = ["solid=Score A"]
        if any(t is not None for t in mus_prop):
            subtitle_parts.append("dashed=Score B")
        if nd["has_bayes"]:
            subtitle_parts.append("◇=Score Bayes")
 
        ax.set_ylim(-0.12, 1.18)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title(
            f"Sensor {sensor_id}  pos={data['pos']:.1f}m  ({len(timeline)} events)"
            f"  — {'   '.join(subtitle_parts)}",
            fontsize=9
        )
        ax.legend(handles=legend_handles, fontsize=6,
                  loc="upper right", framealpha=0.85, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xlim(global_t_lo, global_t_hi)
 
        # Only label x ticks on bottom subplot — shared axis handles the rest
        if si == n_sens - 1:
            raw_ticks = ax.get_xticks()
            ax.set_xticklabels([_to_ts(t) for t in raw_ticks],
                               rotation=25, ha="right", fontsize=7)
            ax.set_xlabel("Timestamp", fontsize=8)
 
    # Shared x-axis label formatting pass for all axes
    fig.align_ylabels(axes[:, 0])
 
    plt.tight_layout()
    out3 = os.path.join(output_dir, "viz3_gaussian_curves.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"[VIZ3] Saved: {out3}")
    plt.show()
 


def viz_ownership_table(sensor_data: dict, vehicles: list,
                        output_dir: str = ".") -> None:
    """
    One table per sensor.
    Always    : Score A column per vehicle
    If Bayes  : Score Bayes column per vehicle alongside Score A

    Compatible with all three scoring functions.
    Saved: <output_dir>/viz4_ownership_table.png
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)

    sensors  = list(sensor_data.keys())
    n_sens   = len(sensors)
    n_veh    = len(vehicles)
    if n_sens == 0:
        print("[VIZ4] No data"); return

    # Check globally whether any sensor has Bayes
    has_bayes_any = any(
        _normalise_sensor_data(sensor_data[s], vehicles)["has_bayes"]
        for s in sensors
    )
    cols_per_veh = 2 if has_bayes_any else 1

    cmap       = cm.get_cmap("tab10", n_veh)
    v_colors   = [cmap(i) for i in range(n_veh)]
    score_cmap = plt.cm.YlOrRd

    row_h = 0.45
    fig_h = sum(max(2.2, len(sensor_data[s]["timeline"]) * row_h + 1.8)
                for s in sensors)
    fig_w = max(12, n_veh * (2.2 * cols_per_veh) + 4)

    fig, axes = plt.subplots(n_sens, 1, figsize=(fig_w, fig_h), squeeze=False)

    title = "Ownership Table — Score A (WIM)"
    if has_bayes_any:
        title += " | Score Bayes (Posterior)"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for si, sensor_id in enumerate(sensors):
        data     = sensor_data[sensor_id]
        timeline = data["timeline"]
        nd       = _normalise_sensor_data(data, vehicles)
        ax       = axes[si][0]
        ax.axis("off")

        n_events = len(timeline)
        if n_events == 0:
            ax.set_title(f"Sensor {sensor_id} — no events", fontsize=9)
            continue

        # Build column headers
        col_labels = ["Timestamp", "Amplitude"]
        for v in vehicles:
            col_labels.append(f"V{v.vehicle_id}\nA")
            if has_bayes_any:
                col_labels.append(f"V{v.vehicle_id}\nBay")

        cell_text   = []
        cell_colors = []

        for ei, event in enumerate(timeline):
            sc_a_row   = nd["scores_wim"][ei]  if nd["scores_wim"]  else [0.0] * n_veh
            sc_bay_row = nd["scores_bayes"][ei] if nd["has_bayes"]   else None

            text_row  = [_to_ts(event.timestamp), f"{event.peak_amplitude:+.5f}"]
            color_row = ["#f0f0f0", "#f0f0f0"]

            for vi in range(n_veh):
                sc_a = sc_a_row[vi]
                text_row.append(f"{sc_a:.3f}")
                rgba_a = list(score_cmap(sc_a)); rgba_a[3] = 0.75
                color_row.append(rgba_a)

                if has_bayes_any:
                    sc_bay = sc_bay_row[vi] if sc_bay_row else 0.0
                    text_row.append(f"{sc_bay:.3f}")
                    rgba_bay = list(score_cmap(sc_bay)); rgba_bay[3] = 0.75
                    color_row.append(rgba_bay)

            cell_text.append(text_row)
            cell_colors.append(color_row)

        tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                       colLabels=col_labels, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.6)

        # Bold black border = Score A owner per row
        for ei in range(n_events):
            sc_a_row  = nd["scores_wim"][ei] if nd["scores_wim"] else [0.0] * n_veh
            best_vi_a = int(np.argmax(sc_a_row))
            col_a     = 2 + best_vi_a * cols_per_veh
            tbl[(ei+1, col_a)].set_edgecolor("black")
            tbl[(ei+1, col_a)].set_linewidth(2.2)

        # Bold colour border = Score Bayes owner per row
        if has_bayes_any:
            for ei in range(n_events):
                sc_bay_row = nd["scores_bayes"][ei] if nd["has_bayes"] else [0.0] * n_veh
                best_vi_y  = int(np.argmax(sc_bay_row))
                col_bay    = 2 + best_vi_y * cols_per_veh + 1
                tbl[(ei+1, col_bay)].set_edgecolor(v_colors[best_vi_y])
                tbl[(ei+1, col_bay)].set_linewidth(2.2)

        # Base header styling
        for ci in range(len(col_labels)):
            tbl[(0, ci)].set_facecolor("#2c3e50")
            tbl[(0, ci)].set_text_props(color="white", fontweight="bold")

        # Vehicle header colour coding
        for vi, v in enumerate(vehicles):
            col_a = 2 + vi * cols_per_veh
            c     = list(v_colors[vi]); c[3] = 1.0
            lum   = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
            txt   = "white" if lum < 0.55 else "black"
            tbl[(0, col_a)].set_facecolor(c)
            tbl[(0, col_a)].set_text_props(color=txt, fontweight="bold")

            if has_bayes_any:
                col_bay = col_a + 1
                c_bay   = [min(x+0.25, 1.0) for x in c[:3]] + [1.0]
                lum_bay = 0.299*c_bay[0] + 0.587*c_bay[1] + 0.114*c_bay[2]
                tbl[(0, col_bay)].set_facecolor(c_bay)
                tbl[(0, col_bay)].set_text_props(
                    color="white" if lum_bay < 0.55 else "black",
                    fontweight="bold"
                )

        subtitle = "A = WIM Gaussian"
        if has_bayes_any:
            subtitle += "   Bay = Bayesian posterior"
        ax.set_title(
            f"Sensor {sensor_id}  pos={data['pos']:.1f}m  ({n_events} events)  | {subtitle}",
            fontsize=9, pad=8, loc="left"
        )

    plt.tight_layout()
    out4 = os.path.join(output_dir, "viz4_ownership_table.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"[VIZ4] Saved: {out4}")
    plt.show()



def viz_vehicle_progression(
    sensor_data      : dict,
    vehicles         : list,
    sensor_positions : dict,
    output_dir       : str = ".",
) -> None:
    """
    Interactive Plotly visualisation of vehicle progression across all
    boundary sensors.

    X axis  : sensor positions in order (labelled by position in metres)
    Y axis  : elapsed seconds since vehicle's first detected timestamp
              above threshold (Y=0 = first actual detection, always >= 0)
              Falls back to t_min_w at sensor 1 if no detection found.

    One line per vehicle, fixed colour per vehicle.
    Points coloured by Score Bayes opacity:
        fully opaque  = high confidence
        faded         = near threshold

    Only plots points where best Score Bayes >= MIN_SCORE_TO_UPDATE.
    Gaps in line where vehicle is below threshold at a sensor.

    Saved: <output_dir>/viz_progression.html  (interactive)
           <output_dir>/viz_progression.png   (static, requires kaleido)
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    import plotly.colors as pc

    os.makedirs(output_dir, exist_ok=True)

    MIN_SCORE_TO_UPDATE = 0.05

    # ── Sensors in position order ─────────────────────────────────────────
    ordered     = sorted(sensor_positions.items(), key=lambda x: x[1])
    sensor_ids  = [s for s, _ in ordered]
    sensor_poss = {s: p for s, p in ordered}

    # X axis uses integer indices for even spacing; labels show position
    x_indices = list(range(len(sensor_ids)))
    x_tick_labels = [f"{sensor_poss[s]:.0f}m" for s in sensor_ids]

    # ── Colour palette ────────────────────────────────────────────────────
    palette     = pc.qualitative.Plotly + pc.qualitative.Dark24
    base_colors = [palette[i % len(palette)] for i in range(len(vehicles))]

    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _score_to_rgba(hex_color, score, min_alpha=0.15, max_alpha=1.0):
        alpha = min_alpha + (max_alpha - min_alpha) * float(score)
        r, g, b = _hex_to_rgb(hex_color)
        return f"rgba({r},{g},{b},{alpha:.2f})"

    # ── Find Y=0 anchor per vehicle ───────────────────────────────────────
    # Anchor = actual first detected timestamp above threshold.
    # Fallback = t_min_w at first sensor (earliest possible arrival).
    first_sensor_pos = sensor_poss[sensor_ids[0]]

    anchors = {}
    for vi, vehicle in enumerate(vehicles):
        # Fallback anchor: earliest possible arrival at sensor 1
        d_wim  = first_sensor_pos - (-3540.0)
        t_min_w = vehicle.entry_ts + d_wim / vehicle.v_max
        fallback = t_min_w

        # Search for actual first detection above threshold
        first_ts = None
        for sensor_id in sensor_ids:
            data = sensor_data.get(sensor_id)
            if data is None or not data.get("timeline"):
                continue
            timeline     = data["timeline"]
            scores_bayes = data.get("scores_bayes", [])
            if not scores_bayes:
                continue
            bay_vals = [
                scores_bayes[ei][vi] if scores_bayes[ei][vi] is not None else 0.0
                for ei in range(len(timeline))
            ]
            best_ei    = int(np.argmax(bay_vals))
            best_bayes = bay_vals[best_ei]
            if best_bayes >= MIN_SCORE_TO_UPDATE:
                ts = timeline[best_ei].timestamp
                if first_ts is None or ts < first_ts:
                    first_ts = ts

        anchors[vehicle.vehicle_id] = first_ts if first_ts is not None else fallback

    # ── Build traces ──────────────────────────────────────────────────────
    fig = go.Figure()

    for vi, vehicle in enumerate(vehicles):
        base_color = base_colors[vi]
        anchor_ts  = anchors[vehicle.vehicle_id]

        x_line   = []   # x indices for line (with None gaps)
        y_line   = []

        x_pts    = []   # x indices for detected points only
        y_pts    = []
        pt_colors = []
        pt_hover  = []
        pt_sizes  = []

        for xi, sensor_id in enumerate(sensor_ids):
            data = sensor_data.get(sensor_id)
            if data is None or not data.get("timeline"):
                x_line.append(xi); y_line.append(None)
                continue

            timeline     = data["timeline"]
            scores_bayes = data.get("scores_bayes", [])
            scores_wim   = data.get("scores_wim", data.get("scores", []))

            if not scores_bayes:
                x_line.append(xi); y_line.append(None)
                continue

            bay_vals = [
                scores_bayes[ei][vi] if scores_bayes[ei][vi] is not None else 0.0
                for ei in range(len(timeline))
            ]
            best_ei    = int(np.argmax(bay_vals))
            best_bayes = bay_vals[best_ei]

            if best_bayes < MIN_SCORE_TO_UPDATE:
                x_line.append(xi); y_line.append(None)
                continue

            best_sc_a  = (
                scores_wim[best_ei][vi]
                if scores_wim and best_ei < len(scores_wim) else 0.0
            )
            best_event = timeline[best_ei]
            elapsed    = best_event.timestamp - anchor_ts
            actual_ts  = _to_ts(best_event.timestamp)
            pos_m      = sensor_poss[sensor_id]

            # Point size scales slightly with confidence
            pt_size = 8 + best_bayes * 8

            x_line.append(xi);    y_line.append(elapsed)
            x_pts.append(xi);     y_pts.append(elapsed)
            pt_colors.append(_score_to_rgba(base_color, best_bayes))
            pt_sizes.append(pt_size)
            pt_hover.append(
                f"<b>V{vehicle.vehicle_id}  {vehicle.v_est*3.6:.0f} km/h</b><br>"
                f"─────────────────────<br>"
                f"Sensor   : {sensor_id}<br>"
                f"Position : {pos_m:.1f} m<br>"
                f"Timestamp: {actual_ts}<br>"
                f"Elapsed  : {elapsed:.2f} s<br>"
                f"Score Bayes : <b>{best_bayes:.4f}</b><br>"
                f"Score A     : {best_sc_a:.4f}"
            )

        if not x_pts:
            continue

        # Connecting line (faint)
        fig.add_trace(go.Scatter(
            x          = x_line,
            y          = y_line,
            mode       = "lines",
            line       = dict(color=base_color, width=1.2),
            opacity    = 0.3,
            showlegend = False,
            hoverinfo  = "skip",
            legendgroup = f"V{vehicle.vehicle_id}",
        ))

        # Detection points coloured by confidence
        fig.add_trace(go.Scatter(
            x    = x_pts,
            y    = y_pts,
            mode = "markers",
            marker = dict(
                color   = pt_colors,
                size    = pt_sizes,
                line    = dict(color=base_color, width=1.2),
                symbol  = "circle",
            ),
            text          = pt_hover,
            hovertemplate = "%{text}<extra></extra>",
            name          = (f"V{vehicle.vehicle_id}  "
                             f"{vehicle.v_est*3.6:.0f} km/h  "
                             f"entry={_to_ts(vehicle.entry_ts)}"),
            legendgroup   = f"V{vehicle.vehicle_id}",
            showlegend    = True,
        ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title = dict(
            text  = ("Vehicle Progression Through Sensors<br>"
                     "<sup>Point opacity = Score Bayes confidence  |  "
                     "Point size = confidence  |  "
                     "Gaps = below threshold</sup>"),
            x     = 0.5,
            font  = dict(size=16, family="Arial"),
        ),
        xaxis = dict(
            tickmode     = "array",
            tickvals     = x_indices,
            ticktext     = x_tick_labels,
            tickangle    = -60,
            tickfont     = dict(size=9),
            title        = dict(text="Sensor Position (m)", font=dict(size=12)),
            showgrid     = True,
            gridcolor    = "rgba(200,200,200,0.4)",
            gridwidth    = 1,
            showline     = True,
            linecolor    = "grey",
            rangeslider  = dict(visible=True, thickness=0.04),
        ),
        yaxis = dict(
            title     = dict(
                text  = "Time Since First Detection (seconds)",
                font  = dict(size=12)
            ),
            showgrid    = True,
            gridcolor   = "rgba(200,200,200,0.4)",
            gridwidth   = 1,
            zeroline    = True,
            zerolinecolor = "rgba(100,100,100,0.6)",
            zerolinewidth = 1.5,
            showline    = True,
            linecolor   = "grey",
        ),
        legend = dict(
            title      = dict(text="Vehicles", font=dict(size=11)),
            font       = dict(size=10),
            itemclick  = "toggle",
            bgcolor    = "rgba(255,255,255,0.85)",
            bordercolor = "lightgrey",
            borderwidth = 1,
        ),
        hovermode    = "closest",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        width        = 1800,
        height       = max(650, len(vehicles) * 35 + 250),
        margin       = dict(l=80, r=40, t=100, b=160),
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_html = os.path.join(output_dir, "viz_progression.html")
    out_png  = os.path.join(output_dir, "viz_progression.png")

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[VIZ-PROG] Saved interactive: {out_html}")

    try:
        fig.write_image(out_png, scale=2)
        print(f"[VIZ-PROG] Saved static:      {out_png}")
    except Exception as e:
        print(f"[VIZ-PROG] Static PNG skipped ({e})  "
              f"→ install kaleido: pip install kaleido")

    fig.show()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Bridge vehicle tracker")
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to save visualisation PNG files (default: output/)"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[OUTPUT] Visualisations will be saved to: {args.output_dir}/")

    sensor_positions = load_sensor_positions(limit=2)
    vehicles         = load_wim_entries()
    sensor_events    = load_sensor_events()

    sensor_positions_all = load_sensor_positions(limit=None)

    # Swap function name to switch scoring method:
    #   _build_sensor_data           — baseline WIM only
    #   _build_sensor_data_propagated — Bayesian propagation
    #   _build_sensor_data_dual      — Score A + Score B + Score Bayes
    sensor_data = _build_sensor_data(sensor_positions_all, sensor_events, vehicles)

#     viz_vehicle_progression(
#     sensor_data      = sensor_data,
#     vehicles         = vehicles,
#     sensor_positions = sensor_positions_all,
#     output_dir       = args.output_dir,
# )

    viz_heatmap(sensor_data, vehicles, output_dir=args.output_dir)
    viz_gaussian_curves(sensor_data, vehicles, output_dir=args.output_dir)
    viz_ownership_table(sensor_data, vehicles, output_dir=args.output_dir)