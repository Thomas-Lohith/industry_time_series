"""
bridge_tracker.py
=================
Probabilistic vehicle tracking system for an instrumented bridge.

Matches WIM entry records to sensor peak-amplitude events using
Gaussian temporal windows and soft multi-track assignment.

Input sources
-------------
sensor positions    : loaded via load_bridge() from bridge_model.py
sensor_events.csv   : wide format — one row per vehicle, one column per
                      sensor (<sensor_id>_dominant_peaks), each cell
                      holds semicolon-separated "timestamp|amplitude" pairs.
wim_entries.csv     : <index>, StartTimeStr, Velocity (km/h), GrossWeight (kg),
                      timedelta_minutes (inter-vehicle gap — ignored)

Paths configured in config.py.
All design decisions documented in bridge_tracker_context.md
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.shared import config
from src.shared.bridge_model import *

# ---------------------------------------------------------------------------
# Tunable parameters — adjust here only, nothing buried in functions
# ---------------------------------------------------------------------------

SIGMA_DIVISOR          = 4.0    # controls width of temporal Gaussian window
THRESHOLD_MIN          = 0.10   # min normalised probability to count as candidate
THRESHOLD_PRUNE        = 0.02   # min cumulative score to keep a track alive
MISSED_PENALTY         = 0.80   # score multiplier per missed sensor
MASS_TOLERANCE_FRAC    = 0.40   # fraction of expected amplitude used as sigma_amp
VELOCITY_TOLERANCE_KMH = 5.0    # ±km/h applied to WIM speed → v_min / v_max
TIMESTAMP_FORMAT       = "%Y-%m-%d %H:%M:%S"   # adjust if format differs

# Sensor column suffix in sensor_events.csv
_SENSOR_COL_SUFFIX = "_dominant_peaks"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensorEvent:
    """One dominant peak recorded at a single sensor within a vehicle window."""
    sensor_id      : str
    timestamp      : float        # seconds from reference epoch (converted at load)
    peak_amplitude : float
    probs          : dict = field(default_factory=dict)  # {track_id: normalised_prob}


@dataclass
class VehicleTrack:
    """
    One vehicle's journey across the bridge.

    matched stores ALL candidates found at each sensor:
        [(sensor_id, [(timestamp, amplitude, probability), ...]), ...]
    Ambiguous sensors (multiple candidates) are fully visible in output.
    """
    track_id  : int
    entry_ts  : float           # seconds from reference epoch (WIM gate time)
    v_min     : float           # m/s
    v_max     : float           # m/s
    mass_kg   : float
    v_est     : float           # current refined velocity estimate (m/s)
    last_ts   : float           # timestamp of current position anchor
    last_pos  : float           # position (m) of current position anchor
    score     : float = 1.0
    matched   : list  = field(default_factory=list)
    status    : str   = "ACTIVE"   # ACTIVE | PRUNED | COMPLETED


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_reference_epoch: Optional[datetime] = None   # set by load_wim_entries() first


def _dt_to_seconds(dt: datetime) -> float:
    """
    Convert a datetime to seconds from the global reference epoch.
    Sets the reference epoch on first call.
    IMPORTANT: load_wim_entries() must be called before load_sensor_events()
    so the epoch is anchored to the first WIM entry, not a sensor event.
    """
    global _reference_epoch
    if _reference_epoch is None:
        _reference_epoch = dt
    return (dt - _reference_epoch).total_seconds()


def _parse_str_to_dt(ts_str: str) -> datetime:
    """
    Parse a timestamp string to a datetime object.
    Tries with microseconds first, falls back to whole seconds.
    """
    for fmt in (TIMESTAMP_FORMAT + ".%f", TIMESTAMP_FORMAT):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse timestamp: '{ts_str}' — check TIMESTAMP_FORMAT constant"
    )


def _parse_timestamp(ts_str: str) -> float:
    """Parse a timestamp string directly to seconds from reference epoch."""
    return _dt_to_seconds(_parse_str_to_dt(ts_str))


# ---------------------------------------------------------------------------
# Physics / probability helpers
# ---------------------------------------------------------------------------

def mass_to_amplitude(mass_kg: float) -> float:
    """
    Placeholder linear calibration.
    Replace with fitted regression when mass-to-amplitude data is available.
    Single function to swap — nothing else in the code changes.
    """
    return 0.005 * mass_kg


def gaussian(x: float, mu: float, sigma: float) -> float:
    """
    Unnormalised Gaussian used as a relative score only.
    Returns values in (0, 1] — does NOT integrate to 1.
    """
    if sigma <= 0:
        return 1.0 if x == mu else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def p_mass_score(amplitude: float, mass_kg: float) -> float:
    """Gaussian score for amplitude consistency with track mass."""
    expected  = mass_to_amplitude(mass_kg)
    sigma_amp = expected * MASS_TOLERANCE_FRAC if expected > 0 else 1.0
    return gaussian(abs(amplitude), expected, sigma_amp)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_sensor_positions() -> dict[str, float]:
    """
    Build Bridge from config paths, extract positions of BOUNDARY sensors only.
    Boundary sensors are those at span junctions (tail + entry of each junction).
    Returns {sensor_id: position_m} for boundary sensors only.

    Uses bridge.find_boundaries() — same method as bridge_model main().
    Non-boundary sensors are ignored entirely by the tracker.
    """
    bridge    = load_bridge(config.position_csv, config.threshold_csv)
    junctions = bridge.find_boundaries()

    positions = {}
    for junction in junctions:
        for sensor in junction.all_sensors():
            positions[sensor.sensor_id] = sensor.distance

    print(f"[LOAD] {len(junctions)} span junctions found")
    print(f"[LOAD] {len(positions)} boundary sensors extracted")
    return positions


def load_wim_entries() -> list[VehicleTrack]:
    """
    Load WIM vehicle entries. MUST be called before load_sensor_events()
    to establish the reference epoch from the first WIM timestamp.

    Returns list[VehicleTrack], one per CSV row.

    CSV columns:
        <unnamed index>    → IGNORED (random pandas row index, e.g. 2654, 4360)
        StartTimeStr       → entry timestamp
        Velocity           → speed in km/h → converted to m/s; ±5 km/h tolerance
        GrossWeight        → mass in kg
        timedelta_minutes  → inter-vehicle time gap, IGNORED

    track_id: sequential integer assigned by row order (0, 1, 2, ...)
              — stable, meaningful, independent of source CSV index.

    Position anchor:
        last_pos = 0.0       — WIM gate is the origin of the position axis.
                               Sensor positions encode full global distance
                               from WIM gate, so no offset is needed here.
        last_ts  = entry_ts  — vehicle is at WIM gate at this timestamp.
                               Travel time to any sensor is computed naturally
                               as sensor_pos / v_est inside the Gaussian.
    """
    tracks = []
    with open(config.WIM_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            if not row.get("Velocity"):   # skip malformed / empty rows
                continue
            speed_kmh = float(row["Velocity"])
            v_min     = (speed_kmh - VELOCITY_TOLERANCE_KMH) / 3.6
            v_max     = (speed_kmh + VELOCITY_TOLERANCE_KMH) / 3.6
            v_est     = speed_kmh / 3.6
            entry_ts  = _parse_timestamp(row["StartTimeStr"])
            # timedelta_minutes — inter-vehicle time gap, not used

            tracks.append(VehicleTrack(
                track_id = idx,          # sequential: 0, 1, 2, ...
                entry_ts = entry_ts,
                v_min    = v_min,
                v_max    = v_max,
                mass_kg  = float(row["GrossWeight"]),
                v_est    = v_est,
                # Anchor: vehicle at WIM gate (pos=0) at WIM entry timestamp.
                # Gaussian naturally computes travel time via sensor_pos / v_est.
                last_ts  = entry_ts,
                last_pos = 0.0,
            ))

    print(f"[LOAD] {len(tracks)} WIM vehicle entries (track_ids 0–{len(tracks)-1})")
    return tracks


def load_sensor_events() -> dict[str, list[SensorEvent]]:
    """
    Load sensor peak events from wide-format CSV.
    Returns {sensor_id: [SensorEvent, ...]} grouped for O(1) lookup.

    MUST be called after load_wim_entries() — epoch is set by WIM loader.

    CSV structure (one row per vehicle):
        vehicle_id, original_timestamp, <sensor_id>_dominant_peaks, ...

    Each sensor cell: semicolon-separated "timestamp|amplitude" pairs.
        e.g. "2025-03-05 19:08:46.700000|0.006926;..."

    Date fix: sensor event timestamps have incorrect dates (recording artifact
    — month/day appear swapped relative to WIM dates). The correct date is
    taken from original_timestamp for each row; only the time is preserved
    from the sensor event timestamp string.
    """
    events:  dict[str, list[SensorEvent]] = {}
    skipped = 0

    with open(config.SENSOR_EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Identify sensor columns by suffix
        sensor_cols = [
            col for col in reader.fieldnames
            if col.endswith(_SENSOR_COL_SUFFIX)
        ]
        print(f"[LOAD] Found {len(sensor_cols)} sensor columns in events CSV")

        for row in reader:
            # Anchor date: take from this vehicle's WIM-matched timestamp
            original_dt  = _parse_str_to_dt(row["original_timestamp"].strip())
            correct_date = original_dt.date()

            for col in sensor_cols:
                sensor_id = col[: -len(_SENSOR_COL_SUFFIX)]   # strip suffix
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

                    # Fix date: replace with correct date from original_timestamp
                    # (sensor timestamps have wrong date due to recording artifact)
                    corrected_dt = event_dt.replace(
                        year  = correct_date.year,
                        month = correct_date.month,
                        day   = correct_date.day,
                    )

                    evt = SensorEvent(
                        sensor_id      = sensor_id,
                        timestamp      = _dt_to_seconds(corrected_dt),
                        peak_amplitude = float(amp_str),
                    )
                    events.setdefault(sensor_id, []).append(evt)

    total = sum(len(v) for v in events.values())
    if skipped:
        print(f"[LOAD] WARNING: {skipped} event entries skipped (unparseable)")
    print(f"[LOAD] {total} sensor events across {len(events)} sensors")
    return events


# ---------------------------------------------------------------------------
# Core tracking functions — unchanged from design
# ---------------------------------------------------------------------------

def compute_raw_probability(track: VehicleTrack,
                             event: SensorEvent,
                             sensor_pos: float) -> float:
    """
    Raw physics-based score for one (track, event) pair.
    p_raw = p_temporal * p_mass

    track.score intentionally excluded — soft assignment reflects sensor
    physics only. Score applied post-normalisation for gating only.
    """
    d = sensor_pos - track.last_pos

    if d <= 0 or track.v_est <= 0:
        return 0.0

    mu      = track.last_ts + d / track.v_est
    t_min   = track.last_ts + d / track.v_max
    t_max   = track.last_ts + d / track.v_min
    sigma_t = (t_max - t_min) / SIGMA_DIVISOR
    if sigma_t <= 0:
        sigma_t = 1e-3   # safety fallback for identical v_min/v_max edge case

    p_temporal = gaussian(event.timestamp, mu, sigma_t)
    p_mass     = p_mass_score(event.peak_amplitude, track.mass_kg)

    return p_temporal * p_mass


def normalise_event_probs(events: list[SensorEvent]) -> None:
    """
    Normalise each event's probs dict so values sum to 1.0 across tracks.
    Modifies events in place.
    """
    for event in events:
        total = sum(event.probs.values())
        if total > 0:
            for tid in event.probs:
                event.probs[tid] /= total


def update_track(track: VehicleTrack,
                 sensor_id: str,
                 sensor_pos: float,
                 events: list[SensorEvent]) -> None:
    """
    Update one track based on all candidate events at this sensor.

    Candidates: events where normalised_prob * track.score >= THRESHOLD_MIN
    ALL candidates logged with probabilities (ambiguity preserved).
    Velocity refined as weighted average across ALL candidates.
    Score updated using MAX candidate probability.
    Position anchor set to highest-probability candidate.
    """
    d = sensor_pos - track.last_pos

    candidates = [
        (evt, evt.probs.get(track.track_id, 0.0))
        for evt in events
        if evt.probs.get(track.track_id, 0.0) * track.score >= THRESHOLD_MIN
    ]

    if candidates:
        # Log ALL candidates — full ambiguity visible in output
        track.matched.append((
            sensor_id,
            [(evt.timestamp, evt.peak_amplitude, p) for evt, p in candidates]
        ))

        # Velocity refinement — weighted average across ALL candidates
        total_weight = sum(p for _, p in candidates)
        v_sum = 0.0
        for evt, p in candidates:
            delta_t = evt.timestamp - track.last_ts
            if delta_t > 0:
                v_measured = d / delta_t
                v_measured = max(track.v_min, min(track.v_max, v_measured))
                v_sum += p * v_measured

        if total_weight > 0 and v_sum > 0:
            v_refined   = v_sum / total_weight
            blend       = min(total_weight, 1.0)
            track.v_est = blend * v_refined + (1.0 - blend) * track.v_est
            track.v_est = max(track.v_min, min(track.v_max, track.v_est))

        # Score — best match drives the update
        max_p        = max(p for _, p in candidates)
        track.score *= max_p

        # Position anchor — highest-probability candidate
        best_evt, _    = max(candidates, key=lambda x: x[1])
        track.last_ts  = best_evt.timestamp
        track.last_pos = sensor_pos

    else:
        # Missed detection — track survives with penalty
        track.score *= MISSED_PENALTY


def prune_tracks(tracks: list[VehicleTrack]) -> None:
    """Mark ACTIVE tracks below THRESHOLD_PRUNE as PRUNED."""
    for track in tracks:
        if track.status == "ACTIVE" and track.score < THRESHOLD_PRUNE:
            track.status = "PRUNED"


# ---------------------------------------------------------------------------
# Main tracker pipeline
# ---------------------------------------------------------------------------

def run_tracker(sensor_positions: dict[str, float],
                sensor_events:    dict[str, list[SensorEvent]],
                tracks:           list[VehicleTrack]) -> list[VehicleTrack]:
    """
    Process each sensor in ascending position order.
    Returns all tracks (COMPLETED + PRUNED) after the last sensor.
    """
    ordered_sensors = sorted(sensor_positions.items(), key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("BRIDGE TRACKER — starting pipeline")
    print(f"  Sensors     : {len(ordered_sensors)}")
    print(f"  WIM entries : {len(tracks)}")
    print(f"  Sensor range: {ordered_sensors[0][1]:.1f}m — "
          f"{ordered_sensors[-1][1]:.1f}m")
    print("=" * 70)

    for sensor_id, sensor_pos in ordered_sensors:
        events = sensor_events.get(sensor_id, [])
        active = [t for t in tracks if t.status == "ACTIVE"]

        print(f"\n--- Sensor {sensor_id}  |  pos={sensor_pos:.1f}m  |"
              f"  events={len(events)}  |  active tracks={len(active)} ---")

        # Guard: no events at this sensor
        if not events:
            print("  [!] No events — applying missed penalty to all active tracks")
            for track in active:
                track.score *= MISSED_PENALTY
            prune_tracks(tracks)
            continue

        # Step 1 — raw physics probability (track.score excluded)
        for track in active:
            for event in events:
                event.probs[track.track_id] = compute_raw_probability(
                    track, event, sensor_pos
                )

        # Step 2 — normalise per event across competing tracks
        #normalise_event_probs(events)

        # Step 3 — print probability table for inspection
        _print_prob_table(events, active)

        # Step 4 — update each track (multi-candidate collection)
        for track in active:
            update_track(track, sensor_id, sensor_pos, events)

        # Step 5 — prune dead tracks
        prune_tracks(tracks)

        # Step 6 — per-track status
        _print_track_status(active)

    # Lifecycle: remaining ACTIVE tracks → COMPLETED
    # Set here in main loop, not inside update_track (separation of concerns)
    for track in tracks:
        if track.status == "ACTIVE":
            track.status = "COMPLETED"

    completed = [t for t in tracks if t.status == "COMPLETED"]
    pruned    = [t for t in tracks if t.status == "PRUNED"]
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE — {len(completed)} completed, {len(pruned)} pruned")
    print("=" * 70)

    return tracks


# ---------------------------------------------------------------------------
# Output / summary
# ---------------------------------------------------------------------------

def summarise(tracks: list[VehicleTrack]) -> None:
    """
    Full per-track summary.
    Shows all matched sensors, all candidates per sensor with probabilities.
    Flags ambiguous sensors (multiple candidates) with ⚠.
    """
    print("\n" + "=" * 70)
    print("TRACK SUMMARY")
    print("=" * 70)

    for track in tracks:
        print(f"\nTrack {track.track_id:>6}  |  status={track.status:<10}"
              f"  |  score={track.score:.4f}"
              f"  |  v_est={track.v_est * 3.6:.1f} km/h"
              f"  |  mass={track.mass_kg:.0f} kg"
              f"  |  sensors matched={len(track.matched)}")

        for sensor_id, candidates in track.matched:
            ambiguous = len(candidates) > 1
            flag      = "  ⚠ AMBIGUOUS" if ambiguous else ""
            print(f"  Sensor {sensor_id}{flag}")
            for ts, amp, p in candidates:
                print(f"    ts={ts:10.3f}s  amp={amp:+.5f}  p={p:.4f}")


# ---------------------------------------------------------------------------
# Console helpers (internal)
# ---------------------------------------------------------------------------

def _print_prob_table(events: list[SensorEvent],
                      active: list[VehicleTrack]) -> None:
    if not active or not events:
        return
    track_ids = [t.track_id for t in active]
    header = f"  {'timestamp':>12}  {'amplitude':>10}  " + \
             "  ".join(f"T{tid:>6}" for tid in track_ids)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for evt in events:
        row  = f"  {evt.timestamp:>12.3f}  {evt.peak_amplitude:>+10.5f}  "
        row += "  ".join(
            f"{evt.probs.get(tid, 0.0):>8.4f}" for tid in track_ids
        )
        print(row)


def _print_track_status(active: list[VehicleTrack]) -> None:
    for track in active:
        print(f"  Track {track.track_id}: score={track.score:.4f}  "
              f"v_est={track.v_est * 3.6:.1f} km/h  "
              f"candidates logged={len(track.matched)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sensor_positions = load_sensor_positions()

    # WIM entries MUST load first — establishes the reference epoch
    # so all timestamps across both files are on the same relative scale.
    tracks        = load_wim_entries()
    sensor_events = load_sensor_events()

    tracks = run_tracker(sensor_positions, sensor_events, tracks)
    summarise(tracks)