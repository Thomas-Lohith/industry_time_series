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

═══════════════════════════════════════════════════════════════════════
CORE TRACKING LOGIC
═══════════════════════════════════════════════════════════════════════

The tracker answers one question at every sensor:
  "Which WIM vehicle caused this sensor event?"

It does this purely from TIMING — no amplitude, no mass constraints.

──────────────────────────────────────────────────────────────────────
STEP 1 — Expected arrival window  (physics)
──────────────────────────────────────────────────────────────────────
For each active track and each sensor event the tracker computes
when the vehicle SHOULD arrive at this sensor.

Given:
  • last_pos  — position (m) where the vehicle was last confirmed
  • last_ts   — time (s) at that position
  • sensor_pos — position (m) of the current sensor
  • v_est     — current best velocity estimate (m/s)
  • v_min/v_max — WIM speed ± VELOCITY_TOLERANCE_KMH (m/s)

Distance to sensor:
  d = sensor_pos − last_pos

Expected arrival time (mean of Gaussian):
  mu = last_ts + d / v_est

Earliest / latest plausible arrival (from speed band):
  t_min = last_ts + d / v_max      (fastest → earliest)
  t_max = last_ts + d / v_min      (slowest → latest)

Temporal uncertainty (Gaussian sigma):
  sigma_t = (t_max − t_min) / SIGMA_DIVISOR
  capped at MAX_SIGMA_T_SECONDS to prevent runaway growth
  after multiple missed sensors

──────────────────────────────────────────────────────────────────────
STEP 2 — Temporal probability score
──────────────────────────────────────────────────────────────────────
  p_temporal = exp( −0.5 × ((event.timestamp − mu) / sigma_t)² )

This is an unnormalised Gaussian (max = 1.0 when event arrives exactly
at mu, falling off symmetrically with timing deviation).

This is the ONLY scoring signal.  Amplitude is not used.

──────────────────────────────────────────────────────────────────────
STEP 3 — Soft multi-track competition (normalisation)
──────────────────────────────────────────────────────────────────────
Multiple WIM vehicles may be on the bridge simultaneously.  Each sensor
event belongs to exactly one vehicle.  After computing p_temporal for
every (track, event) pair at this sensor, the scores are normalised
PER EVENT across competing tracks:

  normalised_prob[track, event] = p_temporal[track, event]
                                  ─────────────────────────
                                  Σ p_temporal[k, event]  over all k

Result: each event's probabilities sum to 1.0 across tracks.
  • An event far from all tracks → all tracks share a low score equally.
  • An event clearly matching one track → that track gets ≈1.0.
  • An event equidistant between two tracks → 0.5 / 0.5 split.

──────────────────────────────────────────────────────────────────────
STEP 4 — Candidate collection and track update
──────────────────────────────────────────────────────────────────────
An event becomes a CANDIDATE for a track if:
  normalised_prob ≥ THRESHOLD_MIN

Multiple candidates per sensor are allowed — all are logged.
This preserves full ambiguity in the output for inspection.

Velocity refinement (weighted average over all candidates):
  v_measured = d / (event.timestamp − last_ts)   [clamped to v_min..v_max]
  v_est = blend × weighted_avg(v_measured) + (1−blend) × v_est
  blend = min(total_candidate_weight, 1.0)

Score update (multiplicative — best candidate wins):
  track.score ×= max(normalised_prob across candidates)

Position anchor update:
  last_ts  = timestamp of highest-probability candidate
  last_pos = sensor_pos

If NO candidate found (miss):
  track.score ×= MISSED_PENALTY

──────────────────────────────────────────────────────────────────────
STEP 5 — Pruning
──────────────────────────────────────────────────────────────────────
After every sensor, tracks with score < THRESHOLD_PRUNE are marked
PRUNED.  They no longer compete for future events.

A track whose score stays above THRESHOLD_PRUNE through all sensors
is marked COMPLETED at the end of the pipeline.

──────────────────────────────────────────────────────────────────────
SENSOR OUTAGE vs GENUINE MISS
──────────────────────────────────────────────────────────────────────
If a sensor reports ZERO events:
  → Hardware outage.  Apply SENSOR_OUTAGE_PENALTY (default 1.0 = none).
  → Do NOT apply MISSED_PENALTY — the vehicle may have passed correctly.

If a sensor has events but NONE match the track:
  → Genuine miss.  Apply MISSED_PENALTY.

──────────────────────────────────────────────────────────────────────
WHY AMPLITUDE IS EXCLUDED
──────────────────────────────────────────────────────────────────────
  1. No lateral side mapping → same vehicle looks "heavy" on near side
     and "light" on far side.  Any absolute amplitude threshold would
     reject valid far-side matches.

  2. mass_to_amplitude calibration is not yet available.  A placeholder
     linear relation introduces systematic bias without adding accuracy.

  3. Bridge dynamics (resonance, multi-axle spread, speed-dependent
     amplification) make amplitude noisy even for identical vehicles.

  Amplitude data is still loaded and stored in SensorEvent.peak_amplitude
  for future use and for manual inspection in the output, but it plays
  NO role in scoring or gating.

═══════════════════════════════════════════════════════════════════════
CHANGE LOG
═══════════════════════════════════════════════════════════════════════
FIX-1  : _parse_str_to_dt() tries both Y-M-D and Y-D-M formats.
FIX-2  : TIMESTAMP_FORMAT removed; _CANDIDATE_FORMATS drives parsing.
FIX-3  : normalise_event_probs() re-enabled (soft multi-track assignment).
FIX-4  : SENSOR_OUTAGE_PENALTY separates hardware outage from miss.
FIX-5  : Velocity refinement warns on delta_t == 0.
FIX-6  : MAX_SIGMA_T_SECONDS caps Gaussian window after multi-miss.
FIX-7  : Geometric-mean score in summarise() avoids multiplicative underflow.
FIX-9  : THRESHOLD_MIN gate uses normalised prob only (not × track.score).
FIX-10 : VELOCITY_TOLERANCE_KMH corrected to ±10 km/h.
DROP-A : All amplitude scoring, bonus, and mass-probability logic removed.
         peak_amplitude kept in SensorEvent for output/inspection only.
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
# Tunable parameters
# ---------------------------------------------------------------------------

SIGMA_DIVISOR          = 4.0    # narrows/widens temporal Gaussian window
THRESHOLD_MIN          = 0.10   # min normalised prob to accept an event as candidate
THRESHOLD_PRUNE        = 0.02   # min cumulative score to keep a track alive
MISSED_PENALTY         = 0.80   # score multiplier on genuine missed detection
SENSOR_OUTAGE_PENALTY  = 1.0    # score multiplier on hardware outage (0 events)
VELOCITY_TOLERANCE_KMH = 10.0   # ±10 km/h band around WIM speed → v_min/v_max
MAX_SIGMA_T_SECONDS    = 30.0   # hard cap on sigma_t (prevents runaway window)

_SENSOR_COL_SUFFIX = "_dominant_peaks"

_CANDIDATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",   # ISO with microseconds   e.g. 2025-03-18 00:17:28.870000
    "%Y-%m-%d %H:%M:%S",       # ISO whole seconds        e.g. 2025-03-18 00:17:28
    "%Y-%d-%m %H:%M:%S.%f",   # swapped day/month + us   e.g. 2025-18-03 00:15:00.000000
    "%Y-%d-%m %H:%M:%S",       # swapped day/month        e.g. 2025-18-03 00:15:00
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensorEvent:
    """
    One dominant peak recorded at a single sensor.
    peak_amplitude is stored for output/inspection but NOT used in scoring.
    """
    sensor_id      : str
    timestamp      : float          # seconds from reference epoch
    peak_amplitude : float          # stored for inspection only
    probs          : dict = field(default_factory=dict)   # {track_id: normalised_prob}


@dataclass
class VehicleTrack:
    """
    One vehicle's journey across the bridge.

    matched: list of per-sensor results
        [(sensor_id, [(timestamp, amplitude, probability), ...]), ...]
    Multiple candidates per sensor are logged to preserve ambiguity.
    """
    track_id  : int
    entry_ts  : float       # seconds from reference epoch (WIM gate time)
    v_min     : float       # m/s  (WIM speed − tolerance)
    v_max     : float       # m/s  (WIM speed + tolerance)
    mass_kg   : float       # stored for output; not used in scoring
    v_est     : float       # current best velocity estimate (m/s)
    last_ts   : float       # timestamp of last confirmed position anchor
    last_pos  : float       # position (m) of last confirmed anchor
    score     : float = 1.0
    matched   : list  = field(default_factory=list)
    status    : str   = "ACTIVE"   # ACTIVE | PRUNED | COMPLETED


# ---------------------------------------------------------------------------
# Timestamp parsing  (FIX-1 / FIX-2)
# ---------------------------------------------------------------------------

_reference_epoch: Optional[datetime] = None


def _dt_to_seconds(dt: datetime) -> float:
    """
    Convert datetime → seconds from the global reference epoch.
    Epoch is fixed to the first datetime seen (always the first WIM entry).
    """
    global _reference_epoch
    if _reference_epoch is None:
        _reference_epoch = dt
    return (dt - _reference_epoch).total_seconds()


def _parse_str_to_dt(ts_str: str) -> datetime:
    """
    FIX-1/FIX-2: Try all _CANDIDATE_FORMATS in order.
    Handles the mixed Y-M-D / Y-D-M formatting found in source CSVs.
    ISO (Y-M-D) is attempted first as the more common convention.
    """
    for fmt in _CANDIDATE_FORMATS:
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse timestamp: '{ts_str}' — tried: {_CANDIDATE_FORMATS}"
    )


def _parse_timestamp(ts_str: str) -> float:
    return _dt_to_seconds(_parse_str_to_dt(ts_str))


# ---------------------------------------------------------------------------
# Gaussian helper
# ---------------------------------------------------------------------------

def gaussian(x: float, mu: float, sigma: float) -> float:
    """
    Unnormalised Gaussian score — returns values in (0, 1].
    Returns 1.0 when x == mu, falls off with distance.
    Used exclusively as a relative temporal score.
    """
    if sigma <= 0:
        return 1.0 if x == mu else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_sensor_positions() -> dict[str, float]:
    """
    Extract positions of BOUNDARY sensors from bridge_model.
    Returns {sensor_id: position_m}.
    Non-boundary sensors are ignored by the tracker.
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
    Load WIM vehicle entries.
    MUST be called before load_sensor_events() — fixes the reference epoch.

    CSV columns used:
        StartTimeStr  → entry timestamp (sets epoch on first call)
        Velocity      → km/h → converted to m/s; ± VELOCITY_TOLERANCE_KMH band
        GrossWeight   → stored on track for output; NOT used in scoring

    Position anchor:
        last_pos = 0.0      (WIM gate is the coordinate origin)
        last_ts  = entry_ts (vehicle is at gate at this moment)
    """
    tracks = []
    with open(config.WIM_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if not row.get("Velocity"):
                continue
            speed_kmh = float(row["Velocity"])
            entry_ts  = _parse_timestamp(row["StartTimeStr"])

            tracks.append(VehicleTrack(
                track_id = idx,
                entry_ts = entry_ts,
                v_min    = (speed_kmh - VELOCITY_TOLERANCE_KMH) / 3.6,
                v_max    = (speed_kmh + VELOCITY_TOLERANCE_KMH) / 3.6,
                mass_kg  = float(row["GrossWeight"]),
                v_est    = speed_kmh / 3.6,
                last_ts  = entry_ts,
                last_pos = 0.0,
            ))

    print(f"[LOAD] {len(tracks)} WIM vehicle entries (track_ids 0–{len(tracks)-1})")
    return tracks


def load_sensor_events() -> dict[str, list[SensorEvent]]:
    """
    Load sensor peak events from wide-format CSV.
    MUST be called after load_wim_entries() — epoch must already be set.

    peak_amplitude is loaded and stored for output/inspection.
    It plays NO role in scoring.

    Date fix: sensor event timestamps may have month/day swapped relative
    to WIM dates (recording artifact).  The correct date is taken from
    original_timestamp for each row; only the time-of-day is preserved
    from the individual sensor event timestamp string.
    """
    events:  dict[str, list[SensorEvent]] = {}
    skipped = 0

    with open(config.SENSOR_EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        sensor_cols = [c for c in reader.fieldnames if c.endswith(_SENSOR_COL_SUFFIX)]
        print(f"[LOAD] Found {len(sensor_cols)} sensor columns in events CSV")

        for row in reader:
            original_dt  = _parse_str_to_dt(row["original_timestamp"].strip())
            correct_date = original_dt.date()

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
                        year  = correct_date.year,
                        month = correct_date.month,
                        day   = correct_date.day,
                    )
                    evt = SensorEvent(
                        sensor_id      = sensor_id,
                        timestamp      = _dt_to_seconds(corrected_dt),
                        peak_amplitude = float(amp_str),   # stored, not scored
                    )
                    events.setdefault(sensor_id, []).append(evt)

    total = sum(len(v) for v in events.values())
    if skipped:
        print(f"[LOAD] WARNING: {skipped} event entries skipped (unparseable)")
    print(f"[LOAD] {total} sensor events across {len(events)} sensors")
    return events


# ---------------------------------------------------------------------------
# Core tracking functions
# ---------------------------------------------------------------------------

def compute_temporal_probability(track:      VehicleTrack,
                                  event:      SensorEvent,
                                  sensor_pos: float) -> float:
    """
    Compute the temporal match score for one (track, event) pair.

    Score = unnormalised Gaussian evaluated at event.timestamp:

        d       = sensor_pos − track.last_pos
        mu      = track.last_ts + d / track.v_est        ← expected arrival
        t_min   = track.last_ts + d / track.v_max        ← fastest plausible
        t_max   = track.last_ts + d / track.v_min        ← slowest plausible
        sigma_t = (t_max − t_min) / SIGMA_DIVISOR        ← spread of window
                  capped at MAX_SIGMA_T_SECONDS

        p = exp( −0.5 × ((event.timestamp − mu) / sigma_t)² )

    Returns 0.0 if the sensor is behind the current track anchor (d ≤ 0)
    or if the track velocity is degenerate (v_est ≤ 0).

    This is the SOLE scoring signal — no amplitude, no mass weighting.
    """
    d = sensor_pos - track.last_pos
    if d <= 0 or track.v_est <= 0:
        return 0.0

    mu      = track.last_ts + d / track.v_est
    t_min   = track.last_ts + d / track.v_max
    t_max   = track.last_ts + d / track.v_min
    sigma_t = min((t_max - t_min) / SIGMA_DIVISOR, MAX_SIGMA_T_SECONDS)

    if sigma_t <= 0:
        sigma_t = 1e-3   # safety fallback: v_min == v_max edge case

    return gaussian(event.timestamp, mu, sigma_t)


def normalise_event_probs(events: list[SensorEvent]) -> None:
    """
    FIX-3: Normalise each event's probs dict so values sum to 1.0 across tracks.

    WHY: Without normalisation every track independently scores the same event
    against its own Gaussian — there is no competition.  Two tracks at similar
    positions could both claim the same event with high probability.
    Normalisation forces them to share:  if track A scores 0.9 and track B
    scores 0.1 on the same event, after normalisation A gets 0.90 and B 0.10.

    Modifies events in-place.
    """
    for event in events:
        total = sum(event.probs.values())
        if total > 0:
            for tid in event.probs:
                event.probs[tid] /= total


def update_track(track:      VehicleTrack,
                 sensor_id:  str,
                 sensor_pos: float,
                 events:     list[SensorEvent]) -> None:
    """
    Update one track from all candidate events at this sensor.

    Candidates: events where normalised_prob ≥ THRESHOLD_MIN  (FIX-9)
    All candidates are logged — ambiguity preserved for inspection.

    Velocity refinement:
        For each candidate compute v_measured = d / delta_t.
        Clamp to [v_min, v_max].
        Blend into v_est: v_est = blend × weighted_avg + (1−blend) × v_est
        blend = min(sum_of_weights, 1.0)  — avoids over-committing on a
        single weak candidate.

    FIX-5: Warn if delta_t == 0 for all candidates (timestamp collision).

    Score:  track.score ×= max(normalised_prob across candidates)
    Anchor: last_ts / last_pos set to highest-probability candidate.

    Miss:   no candidates → track.score ×= MISSED_PENALTY
            last_ts / last_pos intentionally NOT updated on miss — the
            Gaussian for the next sensor is computed from the last CONFIRMED
            position.  sigma_t grows with accumulated distance but is capped
            by MAX_SIGMA_T_SECONDS (FIX-6).
    """
    d = sensor_pos - track.last_pos

    # FIX-9: gate on normalised prob alone — track.score excluded
    candidates = [
        (evt, evt.probs.get(track.track_id, 0.0))
        for evt in events
        if evt.probs.get(track.track_id, 0.0) >= THRESHOLD_MIN
    ]

    if candidates:
        track.matched.append((
            sensor_id,
            [(evt.timestamp, evt.peak_amplitude, p) for evt, p in candidates]
        ))

        total_weight  = sum(p for _, p in candidates)
        v_sum         = 0.0
        zero_dt_count = 0

        for evt, p in candidates:
            delta_t = evt.timestamp - track.last_ts
            if delta_t > 0:
                v_measured = d / delta_t
                v_measured = max(track.v_min, min(track.v_max, v_measured))
                v_sum += p * v_measured
            else:
                zero_dt_count += 1

        if total_weight > 0 and v_sum > 0:
            v_refined   = v_sum / total_weight
            blend       = min(total_weight, 1.0)
            track.v_est = blend * v_refined + (1.0 - blend) * track.v_est
            track.v_est = max(track.v_min, min(track.v_max, track.v_est))
        elif zero_dt_count == len(candidates):
            # FIX-5
            print(f"  [WARN] Track {track.track_id}: velocity not refined at "
                  f"{sensor_id} — delta_t=0 for all {len(candidates)} candidate(s)")

        max_p         = max(p for _, p in candidates)
        track.score  *= max_p

        best_evt, _    = max(candidates, key=lambda x: x[1])
        track.last_ts  = best_evt.timestamp
        track.last_pos = sensor_pos

    else:
        track.score *= MISSED_PENALTY


def prune_tracks(tracks: list[VehicleTrack]) -> None:
    """Mark ACTIVE tracks with score < THRESHOLD_PRUNE as PRUNED."""
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
    Process sensors in ascending position order.

    Per-sensor pipeline:
        1. Compute p_temporal for every (active track, sensor event) pair.
        2. Normalise per event across tracks  → soft competition.
        3. Print probability table.
        4. Update each track  → velocity, score, anchor.
        5. Prune low-score tracks.
        6. Print status lines.

    Sensor outage (zero events):
        Apply SENSOR_OUTAGE_PENALTY (FIX-4) — not MISSED_PENALTY.
        Default 1.0 means no penalty for hardware failure.
    """
    ordered_sensors = sorted(sensor_positions.items(), key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("BRIDGE TRACKER — pure temporal model")
    print(f"  Sensors      : {len(ordered_sensors)}")
    print(f"  WIM entries  : {len(tracks)}")
    print(f"  Speed band   : ± {VELOCITY_TOLERANCE_KMH} km/h")
    print(f"  Max sigma_t  : {MAX_SIGMA_T_SECONDS} s")
    print(f"  Sensor range : {ordered_sensors[0][1]:.1f} m — "
          f"{ordered_sensors[-1][1]:.1f} m")
    print("=" * 70)

    for sensor_id, sensor_pos in ordered_sensors:
        events = sensor_events.get(sensor_id, [])
        active = [t for t in tracks if t.status == "ACTIVE"]

        print(f"\n--- Sensor {sensor_id}  |  pos={sensor_pos:.1f} m  |"
              f"  events={len(events)}  |  active tracks={len(active)} ---")

        # FIX-4: hardware outage — no events, not a vehicle miss
        if not events:
            print(f"  [!] Sensor outage → SENSOR_OUTAGE_PENALTY={SENSOR_OUTAGE_PENALTY}")
            for track in active:
                track.score *= SENSOR_OUTAGE_PENALTY
            prune_tracks(tracks)
            continue

        # Step 1 — temporal probability for every (track, event) pair
        for track in active:
            for event in events:
                event.probs[track.track_id] = compute_temporal_probability(
                    track, event, sensor_pos
                )

        # Step 2 — FIX-3: normalise → tracks compete per event
        #normalise_event_probs(events)

        # Step 3 — console table
        _print_prob_table(events, active)

        # Step 4 — update tracks
        for track in active:
            update_track(track, sensor_id, sensor_pos, events)

        # Step 5 — prune
        prune_tracks(tracks)

        # Step 6 — status
        _print_track_status(active)

    # Mark surviving tracks as COMPLETED
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
    Print full per-track results.

    FIX-7: Geometric-mean score = raw_score ^ (1 / n_sensors_matched).
    This normalises for bridge length — a track matching 20 sensors with
    score 0.7^20 ≈ 0.0008 looks the same quality per-sensor as one
    matching 3 sensors with score 0.7^3 ≈ 0.34.  geo_score = 0.70 in
    both cases.

    peak_amplitude is shown for manual inspection; it is NOT scored.
    """
    print("\n" + "=" * 70)
    print("TRACK SUMMARY")
    print("=" * 70)

    for track in tracks:
        n         = max(len(track.matched), 1)
        geo_score = track.score ** (1.0 / n)

        print(f"\nTrack {track.track_id:>6}  |  status={track.status:<10}"
              f"  |  raw_score={track.score:.4f}"
              f"  |  geo_score={geo_score:.4f}"
              f"  |  v_est={track.v_est * 3.6:.1f} km/h"
              f"  |  mass={track.mass_kg:.0f} kg"
              f"  |  sensors matched={len(track.matched)}")

        for sensor_id, candidates in track.matched:
            flag = "  ⚠ AMBIGUOUS" if len(candidates) > 1 else ""
            print(f"  Sensor {sensor_id}{flag}")
            for ts, amp, p in candidates:
                print(f"    ts={ts:10.3f} s  amp={amp:+.5f}  p={p:.4f}")


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print_prob_table(events: list[SensorEvent],
                      active: list[VehicleTrack]) -> None:
    if not active or not events:
        return
    track_ids = [t.track_id for t in active]
    header = (f"  {'timestamp':>12}  {'amplitude':>10}  "
              + "  ".join(f"T{tid:>6}" for tid in track_ids))
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

    # WIM entries first — sets the reference epoch for all timestamps
    tracks        = load_wim_entries()
    sensor_events = load_sensor_events()

    tracks = run_tracker(sensor_positions, sensor_events, tracks)
    summarise(tracks)