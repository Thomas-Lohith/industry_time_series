#!/usr/bin/env python3
"""
Vehicle Speed Estimation from Bridge Accelerometer Data
========================================================

Estimates vehicle speed by correlating vibration events detected across
multiple spatially distributed sensors mounted along a bridge.

Core Idea:
    A single vehicle generates vibration events that appear sequentially
    on different sensors. By identifying and matching these events across
    sensors and measuring their time delays, vehicle speed can be estimated
    using known inter-sensor distances.

Pipeline:
    1. Load & preprocess: Read sensor data, remove DC offset, apply
       threshold-based window filter (zeros out inactive regions)
    2. Event detection: Detect high-amplitude vibration events per sensor
    3. Event correlation: Match events across consecutive sensors using
       time windows derived from plausible speed ranges
    4. Speed estimation: Compute speed = distance / time_delay
    5. Visualization: Plot detected events, correlations, and speed estimates

Usage:
    python vehicle_speed_estimation.py \
        --path /path/to/data.csv \
        --start_time '2025/03/07 01:05:00' \
        --duration_mins 5 \
        --sensor_group campate1a \
        --distances 0 25 50

    # Or with Parquet file:
    python vehicle_speed_estimation.py \
        --path /path/to/data.parquet \
        --start_time '2025/03/07 01:05:00' \
        --duration_mins 3 \
        --sensor_group campate1b \
        --distances 0 30 60

    # Custom sensor list and distances:
    python vehicle_speed_estimation.py \
        --path /path/to/data.csv \
        --start_time '2025/03/07 01:05:00' \
        --duration_mins 5 \
        --sensors 030911FF_x 030911EF_x 03091200_x \
        --distances 0 25 50

Author: Bridge SHM Project
Date: 2026-03
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import polars as pl
from scipy.ndimage import uniform_filter1d


# =============================================================================
# THRESHOLD-BASED WINDOW FILTER
# =============================================================================

def _get_filtered_mask(
    sensor_series: pd.Series,
    threshold: float,
    sample_period: int,
    pre_trigger_samples: int = 0,
) -> np.ndarray:
    """
    Return a boolean mask.
    True  → sample is inside an active window
    False → sample is outside every active window.

    For each sample whose absolute value exceeds `threshold`, a window of
    `sample_period` samples is opened (extended if further crossings occur).
    Optionally includes `pre_trigger_samples` before the trigger.

    Parameters
    ----------
    sensor_series : pd.Series
        1-D signal values.
    threshold : float
        Amplitude threshold to open a window.
    sample_period : int
        Number of samples to extend the window after the last crossing.
    pre_trigger_samples : int
        Samples before the trigger to include in the window.

    Returns
    -------
    np.ndarray (bool)
        Boolean mask of the same length as sensor_series.
    """
    n    = len(sensor_series)
    mask = np.zeros(n, dtype=bool)
    vals = sensor_series.to_numpy()

    i = 0
    while i < n:
        if np.abs(vals[i]) >= threshold:
            start = max(i - pre_trigger_samples, 0)
            end   = min(i + sample_period, n)
            while i < end:
                if np.abs(vals[i]) >= threshold:
                    end = min(i + sample_period, n)
                i += 1
            mask[start:end] = True
        else:
            i += 1
    return mask


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """All tunable parameters for the speed estimation pipeline."""

    # --- Sampling ---
    sampling_rate_hz: float = 100.0       # 10ms interval = 100 Hz
    dt: float = 0.01                      # seconds per sample

    # --- Preprocessing (threshold-based window filter) ---
    filter_threshold: float = 0.0003      # amplitude threshold to open an active window
    filter_sample_period: int = 200       # samples to extend window after last crossing
    filter_pre_trigger: int = 0           # samples before trigger to include in window

    # --- Event Detection ---
    # Envelope smoothing window (samples) for energy computation
    envelope_window: int = 50             # 0.5s at 100Hz
    # Threshold = mean + threshold_sigma * std of the envelope
    threshold_sigma: float = 3.0
    # Minimum event duration in seconds (reject short spikes)
    min_event_duration_s: float = 0.3
    # Maximum event duration in seconds (reject unrealistically long events)
    max_event_duration_s: float = 5.0
    # Minimum gap between separate events (seconds) - merge closer ones
    min_event_gap_s: float = 0.5
    # Use absolute value of signal for envelope (True) or squared (False)
    use_abs_envelope: bool = True

    # --- Event Correlation ---
    # Plausible vehicle speed range (km/h) for computing search windows
    speed_min_kmh: float = 30.0
    speed_max_kmh: float = 150.0
    # Maximum allowed shape dissimilarity (normalized cross-correlation < this => reject)
    min_shape_similarity: float = 0.3
    # Maximum allowed amplitude ratio difference for matching
    max_amplitude_ratio: float = 5.0
    # Minimum amplitude ratio
    min_amplitude_ratio: float = 0.2

    # --- Speed Estimation ---
    # Reject speed estimates outside this range as outliers
    speed_reject_min_kmh: float = 20.0
    speed_reject_max_kmh: float = 200.0

    # --- Predefined Sensor Groups ---
    # Sensor order follows vehicle travel direction
    # IMPORTANT: User must provide actual inter-sensor distances
    sensor_groups: dict = field(default_factory=lambda: {
        # campate1a: sensors 106 -> 105 -> 104 (3 accelerometers in x-direction)
        'campate1a': {
            'sensors': ['030911FF_x', '030911EF_x', '03091200_x'],
            'distances': None,   # meters between consecutive sensors - USER MUST SET
            'description': 'Campate 1a: sensors 106→105→104'
        },
        # campate1a_extended: includes z-direction and additional sensors
        'campate1a_ext': {
            'sensors': ['030911FF_x', '030911EF_x', '03091200_x',
                        '03091155_z', '03091207_x', '03091119_z'],
            'distances': None,
            'description': 'Campate 1a extended: 6 sensors'
        },
        # campate1b: sensors 53 -> 52 -> 51
        'campate1b': {
            'sensors': ['0309100F_x', '030910F6_x', '0309101E_x'],
            'distances': None,
            'description': 'Campate 1b: sensors 53→52→51'
        },
        # campate2: sensors 99, 100, 101
        'campate2': {
            'sensors': ['030911D2_x', '03091005_x', '0309101F_x'],
            'distances': None,
            'description': 'Campate 2: sensors 99, 100, 101'
        },
        # All campate sensors spread across bridge
        'all_campate': {
            'sensors': ['030911FF_x', '03091017_z', '03091113_x',
                        '0309123B_z', '03091111_z', '03091003_x'],
            'distances': None,
            'description': 'All campate sensors across full bridge'
        }
    })


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str, sensor_columns: list[str],
              start_time: str, duration_mins: float) -> pd.DataFrame:
    """
    Load sensor data from CSV or Parquet, filter to time window.

    Parameters
    ----------
    filepath : str
        Path to .csv or .parquet file.
    sensor_columns : list[str]
        Sensor column names to load.
    start_time : str
        Start time string, e.g. '2025/03/07 01:05:00'.
    duration_mins : float
        Duration in minutes to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'time' (datetime) and sensor columns, filtered to window.
    """
    ext = os.path.splitext(filepath)[1].lower()
    time_col = 'time'
    cols_to_load = [time_col] + sensor_columns

    if ext == '.parquet':
        df_pd = pl.scan_parquet(filepath).select(cols_to_load).collect().to_pandas()
    elif ext == '.csv':
        df_pd = pd.read_csv(filepath, sep=';', usecols=cols_to_load)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .parquet")

    # Parse time
    df_pd['time'] = pd.to_datetime(
        df_pd[time_col],
        format='%Y/%m/%d %H:%M:%S:%f',
        errors='coerce',
        exact=False
    )

    df_pd = df_pd.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

    # Filter to time window
    t_start = pd.to_datetime(start_time)
    t_end = t_start + pd.Timedelta(minutes=duration_mins)
    mask = (df_pd['time'] >= t_start) & (df_pd['time'] <= t_end)
    df_pd = df_pd[mask].reset_index(drop=True)

    print(f"Loaded {len(df_pd):,} samples ({duration_mins} min window)")
    print(f"  Time range: {df_pd['time'].iloc[0]} → {df_pd['time'].iloc[-1]}")
    print(f"  Sensors: {sensor_columns}")

    return df_pd


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_signal(raw: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a single sensor signal:
      1. Remove DC offset (mean subtraction)
      2. Handle NaNs by linear interpolation
      3. Apply threshold-based window filter (zeros out inactive regions)
      4. Compute smoothed envelope for event detection

    Parameters
    ----------
    raw : np.ndarray
        Raw acceleration values.
    config : Config
        Pipeline configuration.

    Returns
    -------
    filtered : np.ndarray
        Threshold-filtered signal (DC removed, inactive regions zeroed).
    envelope : np.ndarray
        Smoothed amplitude envelope for event detection.
    """
    # 1. Remove DC offset
    centered = raw - np.nanmean(raw)

    # 2. Handle NaNs by linear interpolation
    nans = np.isnan(centered)
    if nans.any():
        not_nan = ~nans
        if not_nan.sum() > 10:
            centered[nans] = np.interp(
                np.where(nans)[0],
                np.where(not_nan)[0],
                centered[not_nan]
            )
        else:
            centered[nans] = 0.0

    # 3. Threshold-based window filter
    sensor_series = pd.Series(centered)
    mask = _get_filtered_mask(
        sensor_series,
        threshold=config.filter_threshold,
        sample_period=config.filter_sample_period,
        pre_trigger_samples=config.filter_pre_trigger,
    )
    filtered = centered.copy()
    filtered[~mask] = 0.0

    filtering_ratio = mask.sum() / len(mask)
    print(f"    Filter active ratio: {filtering_ratio * 100:.2f}%  "
          f"({mask.sum()}/{len(mask)} samples)")

    # 4. Compute envelope
    if config.use_abs_envelope:
        env_raw = np.abs(filtered)
    else:
        env_raw = filtered ** 2

    envelope = uniform_filter1d(env_raw, size=config.envelope_window)

    return filtered, envelope


# =============================================================================
# EVENT DETECTION
# =============================================================================

@dataclass
class VibrationEvent:
    """A detected vibration event on a single sensor."""
    sensor: str
    start_idx: int             # sample index of event start
    end_idx: int               # sample index of event end
    peak_idx: int              # sample index of peak amplitude
    start_time: float          # seconds from window start
    end_time: float            # seconds from window start
    peak_time: float           # seconds from window start
    peak_amplitude: float      # peak absolute amplitude
    energy: float              # integrated envelope energy
    duration: float            # seconds
    waveform: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def detect_events(filtered: np.ndarray, envelope: np.ndarray,
                  sensor_name: str, config: Config) -> list[VibrationEvent]:
    """
    Detect vibration events in a single sensor signal using adaptive thresholding.

    Method:
        1. Compute threshold = mean(envelope) + sigma * std(envelope)
        2. Find contiguous regions above threshold
        3. Merge events closer than min_event_gap
        4. Filter by duration constraints
        5. Extract peak info and waveform

    Parameters
    ----------
    filtered : np.ndarray
        Threshold-filtered signal.
    envelope : np.ndarray
        Smoothed amplitude envelope.
    sensor_name : str
        Name of this sensor (for labeling).
    config : Config
        Pipeline configuration.

    Returns
    -------
    list[VibrationEvent]
        Detected events sorted by peak time.
    """
    dt = config.dt
    min_samples = int(config.min_event_duration_s / dt)
    max_samples = int(config.max_event_duration_s / dt)
    min_gap_samples = int(config.min_event_gap_s / dt)

    # Adaptive threshold
    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    threshold = env_mean + config.threshold_sigma * env_std

    # Find above-threshold regions
    above = envelope > threshold
    if not above.any():
        return []

    # Find contiguous segments
    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    if len(starts) == 0 or len(ends) == 0:
        return []

    # Pair starts and ends
    n_events = min(len(starts), len(ends))
    segments = list(zip(starts[:n_events], ends[:n_events]))

    # Merge close segments
    merged = []
    if segments:
        current_start, current_end = segments[0]
        for s, e in segments[1:]:
            if s - current_end < min_gap_samples:
                current_end = e  # merge
            else:
                merged.append((current_start, current_end))
                current_start, current_end = s, e
        merged.append((current_start, current_end))

    # Build events with duration filtering
    events = []
    for s, e in merged:
        duration_samples = e - s
        if duration_samples < min_samples:
            continue
        if duration_samples > max_samples:
            continue

        # Find peak within this event
        event_env = envelope[s:e]
        peak_local = np.argmax(event_env)
        peak_idx = s + peak_local

        # Extract waveform (with small padding)
        pad = int(0.1 / dt)  # 100ms padding
        wave_start = max(0, s - pad)
        wave_end = min(len(filtered), e + pad)
        waveform = filtered[wave_start:wave_end].copy()

        event = VibrationEvent(
            sensor=sensor_name,
            start_idx=s,
            end_idx=e,
            peak_idx=peak_idx,
            start_time=s * dt,
            end_time=e * dt,
            peak_time=peak_idx * dt,
            peak_amplitude=np.abs(filtered[peak_idx]),
            energy=np.sum(envelope[s:e]) * dt,
            duration=(e - s) * dt,
            waveform=waveform
        )
        events.append(event)

    events.sort(key=lambda ev: ev.peak_time)
    return events


# =============================================================================
# EVENT CORRELATION & SPEED ESTIMATION
# =============================================================================

@dataclass
class VehiclePass:
    """A matched vehicle passage across multiple sensors."""
    events: list[VibrationEvent]        # one event per sensor, in sensor order
    time_delays: list[float]            # delays between consecutive sensors (seconds)
    speeds: list[float]                 # speed from each consecutive pair (km/h)
    mean_speed_kmh: float               # average speed estimate
    std_speed_kmh: float                # speed variation across pairs
    confidence: float                   # matching confidence (0-1)
    correlation_scores: list[float]     # shape similarity scores


def compute_time_delay_cross_correlation(
    sig1: np.ndarray, sig2: np.ndarray,
    max_lag_samples: int, dt: float
) -> tuple[float, float]:
    """
    Compute time delay between two signals using cross-correlation.

    Uses normalized cross-correlation to find the lag that maximizes
    similarity. This is more robust than simple peak-to-peak timing.

    Parameters
    ----------
    sig1, sig2 : np.ndarray
        Two signal segments to correlate.
    max_lag_samples : int
        Maximum lag to search (derived from speed range).
    dt : float
        Time step in seconds.

    Returns
    -------
    delay_s : float
        Time delay in seconds (positive = sig2 is later).
    corr_score : float
        Normalized correlation score at best lag (0-1).
    """
    # Normalize signals
    s1 = sig1 - np.mean(sig1)
    s2 = sig2 - np.mean(sig2)

    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0, 0.0

    # Full cross-correlation
    corr = np.correlate(s1, s2, mode='full')
    corr = corr / (norm1 * norm2)

    # Restrict to valid lag range
    mid = len(s1) - 1
    lag_start = max(0, mid - max_lag_samples)
    lag_end = min(len(corr), mid + max_lag_samples + 1)
    corr_window = corr[lag_start:lag_end]

    if len(corr_window) == 0:
        return 0.0, 0.0

    best_local = np.argmax(corr_window)
    best_idx = lag_start + best_local
    best_lag = best_idx - mid
    best_score = corr_window[best_local]

    delay_s = best_lag * dt
    return delay_s, float(best_score)


def correlate_events_across_sensors(
    all_events: dict[str, list[VibrationEvent]],
    sensor_order: list[str],
    distances: list[float],
    filtered_signals: dict[str, np.ndarray],
    config: Config
) -> list[VehiclePass]:
    """
    Match vibration events across consecutive sensors to identify vehicle passages.

    Strategy:
        For each event on the first sensor, search for matching events on
        subsequent sensors within time windows derived from the plausible
        speed range and known inter-sensor distances.

    Two-stage matching:
        1. Coarse: Use peak time windows from speed range constraints
        2. Fine: Refine with cross-correlation of full signal segments

    Parameters
    ----------
    all_events : dict
        {sensor_name: [VibrationEvent, ...]} for each sensor.
    sensor_order : list[str]
        Sensors in order of vehicle travel direction.
    distances : list[float]
        Cumulative distances from first sensor (meters). Length = len(sensor_order).
    filtered_signals : dict
        {sensor_name: np.ndarray} filtered signals for cross-correlation.
    config : Config
        Pipeline configuration.

    Returns
    -------
    list[VehiclePass]
        Matched vehicle passages with speed estimates.
    """
    dt = config.dt
    v_min = config.speed_min_kmh / 3.6   # m/s
    v_max = config.speed_max_kmh / 3.6   # m/s

    if len(sensor_order) < 2:
        print("  Need at least 2 sensors for correlation")
        return []

    vehicle_passes = []
    first_sensor = sensor_order[0]
    first_events = all_events.get(first_sensor, [])

    if not first_events:
        print(f"  No events detected on first sensor: {first_sensor}")
        return []

    # Track which events on downstream sensors have been matched
    used_events = {s: set() for s in sensor_order[1:]}

    for anchor_event in first_events:
        matched_chain = [anchor_event]
        delays = []
        corr_scores = []
        speeds_pair = []
        valid_chain = True

        for i in range(1, len(sensor_order)):
            next_sensor = sensor_order[i]
            next_events = all_events.get(next_sensor, [])

            # Compute expected time window
            # Distance from previous sensor
            d = distances[i] - distances[i - 1]  # meters
            if d <= 0:
                print(f"  Warning: non-positive distance between {sensor_order[i-1]} "
                      f"and {next_sensor}: {d}m")
                valid_chain = False
                break

            # Time window: [d/v_max, d/v_min] from the previous event's peak
            prev_event = matched_chain[-1]
            t_min_delay = d / v_max  # fastest vehicle → smallest delay
            t_max_delay = d / v_min  # slowest vehicle → largest delay

            expected_t_min = prev_event.peak_time + t_min_delay
            expected_t_max = prev_event.peak_time + t_max_delay

            # Find candidate events within the time window
            candidates = []
            for j, ev in enumerate(next_events):
                if j in used_events[next_sensor]:
                    continue
                if expected_t_min <= ev.peak_time <= expected_t_max:
                    # Check amplitude compatibility
                    amp_ratio = ev.peak_amplitude / (anchor_event.peak_amplitude + 1e-10)
                    if config.min_amplitude_ratio <= amp_ratio <= config.max_amplitude_ratio:
                        candidates.append((j, ev))

            if not candidates:
                valid_chain = False
                break

            # Refine with cross-correlation
            prev_ev = matched_chain[-1]
            seg_half = int(max(prev_ev.duration, 0.5) / dt)
            seg1_start = max(0, prev_ev.peak_idx - seg_half)
            seg1_end = min(len(filtered_signals[prev_ev.sensor]),
                          prev_ev.peak_idx + seg_half)
            seg1 = filtered_signals[prev_ev.sensor][seg1_start:seg1_end]

            max_lag = int(t_max_delay / dt) + seg_half

            best_match = None
            best_corr = -1.0
            best_delay = 0.0

            for j, cand_ev in candidates:
                seg2_start = max(0, cand_ev.peak_idx - seg_half)
                seg2_end = min(len(filtered_signals[next_sensor]),
                              cand_ev.peak_idx + seg_half)
                seg2 = filtered_signals[next_sensor][seg2_start:seg2_end]

                min_len = min(len(seg1), len(seg2))
                if min_len < 10:
                    continue

                delay_cc, score = compute_time_delay_cross_correlation(
                    seg1[:min_len], seg2[:min_len], max_lag, dt
                )

                if score > best_corr and score >= config.min_shape_similarity:
                    best_corr = score
                    best_match = (j, cand_ev)
                    best_delay = cand_ev.peak_time - prev_ev.peak_time

            if best_match is None:
                valid_chain = False
                break

            j, matched_ev = best_match
            used_events[next_sensor].add(j)
            matched_chain.append(matched_ev)
            delays.append(best_delay)
            corr_scores.append(best_corr)

            if best_delay > 0:
                speed_ms = d / best_delay
                speed_kmh = speed_ms * 3.6
            else:
                speed_kmh = float('inf')
            speeds_pair.append(speed_kmh)

        if valid_chain and len(matched_chain) >= 2:
            valid_speeds = [s for s in speeds_pair
                           if config.speed_reject_min_kmh <= s <= config.speed_reject_max_kmh]

            if valid_speeds:
                mean_speed = np.mean(valid_speeds)
                std_speed = np.std(valid_speeds) if len(valid_speeds) > 1 else 0.0

                completeness = len(matched_chain) / len(sensor_order)
                avg_corr = np.mean(corr_scores) if corr_scores else 0.0
                speed_consistency = 1.0 / (1.0 + std_speed / (mean_speed + 1e-10))
                confidence = (completeness * 0.3 + avg_corr * 0.4 +
                             speed_consistency * 0.3)

                vp = VehiclePass(
                    events=matched_chain,
                    time_delays=delays,
                    speeds=speeds_pair,
                    mean_speed_kmh=mean_speed,
                    std_speed_kmh=std_speed,
                    confidence=confidence,
                    correlation_scores=corr_scores
                )
                vehicle_passes.append(vp)

    vehicle_passes.sort(key=lambda vp: vp.events[0].peak_time)
    return vehicle_passes


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_detection_results(
    df: pd.DataFrame,
    filtered_signals: dict[str, np.ndarray],
    envelopes: dict[str, np.ndarray],
    all_events: dict[str, list[VibrationEvent]],
    sensor_order: list[str],
    config: Config,
    save_path: str = 'graphs/event_detection.png'
):
    """
    Plot filtered signals with detected events highlighted for each sensor.
    """
    n_sensors = len(sensor_order)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3.5 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    time_s = np.arange(len(filtered_signals[sensor_order[0]])) * config.dt
    colors = plt.cm.Set1(np.linspace(0, 1, max(10, n_sensors)))

    for i, sensor in enumerate(sensor_order):
        ax = axes[i]
        sig = filtered_signals[sensor]
        env = envelopes[sensor]
        events = all_events.get(sensor, [])

        # Threshold line
        env_mean = np.mean(env)
        env_std = np.std(env)
        threshold = env_mean + config.threshold_sigma * env_std

        # Plot signal
        ax.plot(time_s, sig, color=colors[i], alpha=0.6, linewidth=0.5,
                label=f'{sensor} (filtered)')
        # Plot envelope
        ax.plot(time_s, env, color='black', alpha=0.8, linewidth=1.0,
                label='Envelope')
        ax.plot(time_s, -env, color='black', alpha=0.8, linewidth=1.0)
        # Threshold
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5,
                    linewidth=0.8, label=f'Threshold ({config.threshold_sigma}σ)')
        ax.axhline(-threshold, color='red', linestyle='--', alpha=0.5,
                    linewidth=0.8)

        # Highlight detected events
        for ev in events:
            ax.axvspan(ev.start_time, ev.end_time,
                      alpha=0.2, color='orange')
            ax.plot(ev.peak_time, sig[ev.peak_idx], 'rv',
                    markersize=8, zorder=5)

        ax.set_ylabel(f'{sensor}\nAccel.', fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_title(f'Sensor: {sensor} — {len(events)} events detected',
                     fontsize=10, fontweight='bold')

    axes[-1].set_xlabel('Time (seconds from window start)', fontsize=11)
    fig.suptitle('Event Detection Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_results(
    vehicle_passes: list[VehiclePass],
    sensor_order: list[str],
    distances: list[float],
    config: Config,
    save_path: str = 'graphs/speed_estimation.png'
):
    """
    Plot matched vehicle passages showing:
      - Top: Time-distance diagram (sensor vs time, with vehicle tracks)
      - Bottom: Speed histogram
    """
    if not vehicle_passes:
        print("  No vehicle passes to plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1]})

    # --- Top: Time-Distance diagram ---
    ax = axes[0]
    cmap = plt.cm.viridis
    pass_colors = cmap(np.linspace(0.2, 0.9, len(vehicle_passes)))

    for k, vp in enumerate(vehicle_passes):
        times = [ev.peak_time for ev in vp.events]
        dists = []
        for ev in vp.events:
            idx = sensor_order.index(ev.sensor)
            dists.append(distances[idx])

        color = pass_colors[k]
        ax.plot(times, dists, 'o-', color=color, linewidth=2,
                markersize=8, alpha=0.8,
                label=f'V{k+1}: {vp.mean_speed_kmh:.0f}±{vp.std_speed_kmh:.0f} km/h '
                      f'(conf={vp.confidence:.2f})')

        mid_idx = len(times) // 2
        ax.annotate(f'{vp.mean_speed_kmh:.0f} km/h',
                    xy=(times[mid_idx], dists[mid_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

    for i, (sensor, d) in enumerate(zip(sensor_order, distances)):
        ax.axhline(d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else 0,
                d, f'  {sensor}', fontsize=7, va='bottom', color='gray')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Distance along bridge (m)', fontsize=11)
    ax.set_title('Vehicle Tracks: Time-Distance Diagram', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Bottom: Speed histogram ---
    ax2 = axes[1]
    all_speeds = [vp.mean_speed_kmh for vp in vehicle_passes]

    if all_speeds:
        bins = np.arange(0, max(all_speeds) + 10, 5)
        ax2.hist(all_speeds, bins=bins, color='steelblue', edgecolor='white',
                 alpha=0.8)
        mean_v = np.mean(all_speeds)
        med_v = np.median(all_speeds)
        ax2.axvline(mean_v, color='red', linestyle='-', linewidth=2,
                     label=f'Mean: {mean_v:.1f} km/h')
        ax2.axvline(med_v, color='orange', linestyle='--', linewidth=2,
                     label=f'Median: {med_v:.1f} km/h')

    ax2.set_xlabel('Speed (km/h)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Speed Distribution ({len(vehicle_passes)} vehicles detected)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_matched_waveforms(
    vehicle_passes: list[VehiclePass],
    filtered_signals: dict[str, np.ndarray],
    config: Config,
    max_passes: int = 6,
    save_path: str = 'graphs/matched_waveforms.png'
):
    """
    Plot aligned waveforms for matched vehicle passes to visually validate
    the event correlation quality.
    """
    n_plot = min(len(vehicle_passes), max_passes)
    if n_plot == 0:
        return

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for k in range(n_plot):
        vp = vehicle_passes[k]
        ax = axes[k]

        for j, ev in enumerate(vp.events):
            half_win = int(max(ev.duration, 0.5) / config.dt)
            start = max(0, ev.peak_idx - half_win)
            end = min(len(filtered_signals[ev.sensor]), ev.peak_idx + half_win)
            wave = filtered_signals[ev.sensor][start:end]

            t = (np.arange(len(wave)) - (ev.peak_idx - start)) * config.dt

            ax.plot(t, wave, color=colors[j % 10], linewidth=1.2,
                    alpha=0.8, label=ev.sensor)

        ax.set_ylabel('Acceleration', fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_title(
            f'Vehicle {k+1}: {vp.mean_speed_kmh:.1f} km/h | '
            f'Confidence: {vp.confidence:.2f} | '
            f'Delays: {[f"{d:.3f}s" for d in vp.time_delays]}',
            fontsize=9, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time relative to peak (seconds)', fontsize=11)
    fig.suptitle('Matched Waveforms (aligned to peak)', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# SUMMARY & REPORTING
# =============================================================================

def print_summary(
    vehicle_passes: list[VehiclePass],
    all_events: dict[str, list[VibrationEvent]],
    sensor_order: list[str],
    distances: list[float]
):
    """Print a detailed summary of detection and speed estimation results."""
    print("\n" + "=" * 70)
    print("  VEHICLE SPEED ESTIMATION — RESULTS SUMMARY")
    print("=" * 70)

    # Event detection summary
    print("\n--- Event Detection ---")
    for sensor in sensor_order:
        events = all_events.get(sensor, [])
        if events:
            amps = [ev.peak_amplitude for ev in events]
            durs = [ev.duration for ev in events]
            print(f"  {sensor}: {len(events)} events | "
                  f"Amp: [{min(amps):.4f}, {max(amps):.4f}] | "
                  f"Dur: [{min(durs):.2f}s, {max(durs):.2f}s]")
        else:
            print(f"  {sensor}: 0 events")

    # Sensor geometry
    print("\n--- Sensor Geometry ---")
    for i, (sensor, d) in enumerate(zip(sensor_order, distances)):
        if i > 0:
            gap = distances[i] - distances[i - 1]
            print(f"  {sensor}: {d:.1f}m (gap: {gap:.1f}m from previous)")
        else:
            print(f"  {sensor}: {d:.1f}m (reference)")

    # Vehicle pass summary
    print(f"\n--- Vehicle Passes: {len(vehicle_passes)} matched ---")
    if vehicle_passes:
        for k, vp in enumerate(vehicle_passes):
            sensors_hit = [ev.sensor for ev in vp.events]
            print(f"\n  Vehicle {k+1}:")
            print(f"    Time (1st sensor): {vp.events[0].peak_time:.2f}s")
            print(f"    Sensors matched:   {' → '.join(sensors_hit)}")
            print(f"    Time delays:       {[f'{d:.3f}s' for d in vp.time_delays]}")
            print(f"    Pair speeds:       {[f'{s:.1f} km/h' for s in vp.speeds]}")
            print(f"    Mean speed:        {vp.mean_speed_kmh:.1f} ± {vp.std_speed_kmh:.1f} km/h")
            print(f"    Confidence:        {vp.confidence:.3f}")
            print(f"    Corr. scores:      {[f'{c:.3f}' for c in vp.correlation_scores]}")

        speeds = [vp.mean_speed_kmh for vp in vehicle_passes]
        confs = [vp.confidence for vp in vehicle_passes]
        print(f"\n--- Overall Statistics ---")
        print(f"  Vehicles detected:   {len(vehicle_passes)}")
        print(f"  Speed range:         {min(speeds):.1f} – {max(speeds):.1f} km/h")
        print(f"  Mean speed:          {np.mean(speeds):.1f} ± {np.std(speeds):.1f} km/h")
        print(f"  Median speed:        {np.median(speeds):.1f} km/h")
        print(f"  Mean confidence:     {np.mean(confs):.3f}")
        print(f"  High-confidence (>0.6): {sum(1 for c in confs if c > 0.6)}")
    else:
        print("  No vehicle passes detected.")
        print("  Try: lower threshold_sigma, lower filter_threshold, or longer time window.")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    filepath: str,
    start_time: str,
    duration_mins: float,
    sensor_order: list[str],
    distances: list[float],
    config: Config,
    output_dir: str = 'graphs'
) -> tuple[list[VehiclePass], dict]:
    """
    Run the complete vehicle speed estimation pipeline.

    Parameters
    ----------
    filepath : str
        Path to data file (CSV or Parquet).
    start_time : str
        Window start time.
    duration_mins : float
        Window duration in minutes.
    sensor_order : list[str]
        Sensors in vehicle travel direction.
    distances : list[float]
        Cumulative distance of each sensor from the first (meters).
    config : Config
        Pipeline configuration.
    output_dir : str
        Directory for output plots.

    Returns
    -------
    vehicle_passes : list[VehiclePass]
        Detected and matched vehicle passages.
    diagnostics : dict
        Intermediate results for further analysis.
    """
    print("=" * 70)
    print("  VEHICLE SPEED ESTIMATION PIPELINE")
    print("=" * 70)
    print(f"  File:          {filepath}")
    print(f"  Window:        {start_time} + {duration_mins} min")
    print(f"  Sensors:       {sensor_order}")
    print(f"  Distances:     {distances} m")
    print(f"  Speed range:   {config.speed_min_kmh}–{config.speed_max_kmh} km/h")
    print(f"  Filter thresh: {config.filter_threshold}  "
          f"sample_period: {config.filter_sample_period}  "
          f"pre_trigger: {config.filter_pre_trigger}")
    print()

    # Validate distances
    assert len(distances) == len(sensor_order), \
        f"distances ({len(distances)}) must match sensor_order ({len(sensor_order)})"
    assert distances[0] == 0, "First distance should be 0 (reference sensor)"
    for i in range(1, len(distances)):
        assert distances[i] > distances[i-1], \
            f"Distances must be strictly increasing: {distances}"

    # Step 1: Load data
    print("Step 1: Loading data...")
    df = load_data(filepath, sensor_order, start_time, duration_mins)

    # Step 2: Preprocess each sensor
    print("\nStep 2: Preprocessing signals (threshold-based window filter)...")
    filtered_signals = {}
    envelopes = {}
    for sensor in sensor_order:
        raw = df[sensor].values.astype(np.float64)
        print(f"  {sensor}: raw range [{np.nanmin(raw):.4f}, {np.nanmax(raw):.4f}]")
        filt, env = preprocess_signal(raw, config)
        filtered_signals[sensor] = filt
        envelopes[sensor] = env
        print(f"    → filtered range [{filt.min():.4f}, {filt.max():.4f}]")

    # Step 3: Detect events per sensor
    print("\nStep 3: Detecting events...")
    all_events = {}
    for sensor in sensor_order:
        events = detect_events(
            filtered_signals[sensor],
            envelopes[sensor],
            sensor, config
        )
        all_events[sensor] = events
        print(f"  {sensor}: {len(events)} events detected")

    # Step 4: Correlate events & estimate speed
    print("\nStep 4: Correlating events across sensors...")
    vehicle_passes = correlate_events_across_sensors(
        all_events, sensor_order, distances,
        filtered_signals, config
    )

    # Step 5: Visualize
    print("\nStep 5: Generating visualizations...")
    plot_detection_results(
        df, filtered_signals, envelopes, all_events,
        sensor_order, config,
        save_path=os.path.join(output_dir, 'event_detection.png')
    )
    plot_correlation_results(
        vehicle_passes, sensor_order, distances, config,
        save_path=os.path.join(output_dir, 'speed_estimation.png')
    )
    plot_matched_waveforms(
        vehicle_passes, filtered_signals, config,
        save_path=os.path.join(output_dir, 'matched_waveforms.png')
    )

    # Step 6: Summary
    print_summary(vehicle_passes, all_events, sensor_order, distances)

    diagnostics = {
        'df': df,
        'filtered_signals': filtered_signals,
        'envelopes': envelopes,
        'all_events': all_events,
        'config': config,
    }

    return vehicle_passes, diagnostics


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate vehicle speed from bridge accelerometer data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined sensor group:
  python vehicle_speed_estimation.py \\
      --path data.csv \\
      --start_time '2025/03/07 01:05:00' \\
      --duration_mins 5 \\
      --sensor_group campate1a \\
      --distances 0 25 50

  # Custom sensors and distances:
  python vehicle_speed_estimation.py \\
      --path data.parquet \\
      --start_time '2025/03/07 01:05:00' \\
      --duration_mins 3 \\
      --sensors 030911FF_x 030911EF_x 03091200_x \\
      --distances 0 25 50

  # Adjust filter sensitivity:
  python vehicle_speed_estimation.py \\
      --path data.csv \\
      --start_time '2025/03/07 01:05:00' \\
      --duration_mins 5 \\
      --sensor_group campate1b \\
      --distances 0 30 60 \\
      --filter_threshold 0.0005 \\
      --filter_sample_period 300 \\
      --threshold_sigma 2.5 \\
      --speed_min 40 --speed_max 130
        """
    )

    # Required
    parser.add_argument('--path', required=True,
                        help='Path to CSV or Parquet data file')
    parser.add_argument('--start_time', required=True,
                        help="Window start time, e.g. '2025/03/07 01:05:00'")
    parser.add_argument('--duration_mins', type=float, required=True,
                        help='Window duration in minutes')

    # Sensor selection (either group or custom list)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sensor_group',
                       choices=['campate1a', 'campate1a_ext', 'campate1b',
                                'campate2', 'all_campate'],
                       help='Predefined sensor group')
    group.add_argument('--sensors', nargs='+',
                       help='Custom sensor column names in travel order')

    # Distances (always required for speed calculation)
    parser.add_argument('--distances', nargs='+', type=float, required=True,
                        help='Cumulative distance (m) of each sensor from first sensor. '
                             'Must start with 0. E.g.: 0 25 50')

    # Filter parameters
    parser.add_argument('--filter_threshold', type=float, default=0.0003,
                        help='Amplitude threshold to open active window (default: 0.0003)')
    parser.add_argument('--filter_sample_period', type=int, default=200,
                        help='Samples to extend window after last threshold crossing (default: 200)')
    parser.add_argument('--filter_pre_trigger', type=int, default=0,
                        help='Samples before trigger to include in window (default: 0)')

    # Tunable parameters
    parser.add_argument('--threshold_sigma', type=float, default=3.0,
                        help='Event detection threshold = mean + sigma*std (default: 3.0)')
    parser.add_argument('--speed_min', type=float, default=30.0,
                        help='Min plausible speed km/h (default: 30)')
    parser.add_argument('--speed_max', type=float, default=150.0,
                        help='Max plausible speed km/h (default: 150)')
    parser.add_argument('--min_event_duration', type=float, default=0.3,
                        help='Min event duration seconds (default: 0.3)')
    parser.add_argument('--min_shape_similarity', type=float, default=0.3,
                        help='Min cross-correlation score for matching (default: 0.3)')
    parser.add_argument('--output_dir', default='graphs',
                        help='Output directory for plots (default: graphs)')

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()

    # Filter parameters
    config.filter_threshold    = args.filter_threshold
    config.filter_sample_period = args.filter_sample_period
    config.filter_pre_trigger  = args.filter_pre_trigger

    # Detection / correlation parameters
    config.threshold_sigma        = args.threshold_sigma
    config.speed_min_kmh          = args.speed_min
    config.speed_max_kmh          = args.speed_max
    config.min_event_duration_s   = args.min_event_duration
    config.min_shape_similarity   = args.min_shape_similarity

    # Resolve sensors
    if args.sensor_group:
        group_info = config.sensor_groups[args.sensor_group]
        sensor_order = group_info['sensors']
        print(f"Using sensor group: {args.sensor_group} — {group_info['description']}")
    else:
        sensor_order = args.sensors

    distances = args.distances

    # Validate distances match sensors
    if len(distances) != len(sensor_order):
        print(f"ERROR: {len(distances)} distances provided but "
              f"{len(sensor_order)} sensors selected.")
        print(f"  Sensors:   {sensor_order}")
        print(f"  Distances: {distances}")
        sys.exit(1)

    vehicle_passes, diagnostics = run_pipeline(
        filepath=args.path,
        start_time=args.start_time,
        duration_mins=args.duration_mins,
        sensor_order=sensor_order,
        distances=distances,
        config=config,
        output_dir=args.output_dir
    )

    return vehicle_passes, diagnostics


if __name__ == '__main__':
    main()