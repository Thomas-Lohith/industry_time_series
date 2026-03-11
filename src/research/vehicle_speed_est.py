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
    1. Load data: Read sensor data, remove DC offset
    2. Event extraction: Threshold-triggered windowed extraction per sensor
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


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """All tunable parameters for the speed estimation pipeline."""

    # --- Sampling ---
    sampling_rate_hz: float = 100.0       # 10ms interval = 100 Hz
    dt: float = 0.01                      # seconds per sample

    # --- Event Extraction (threshold-triggered windowing) ---
    # Amplitude threshold: when |signal| crosses this, an event window opens
    event_threshold: float = 0.001
    # Duration of the event window after trigger (seconds)
    event_window_sec: float = 20.0
    # Pre-trigger time to include before the first threshold crossing (seconds)
    event_pre_trigger_sec: float = 5.0

    # --- Minimum / maximum event duration for filtering ---
    min_event_duration_s: float = 0.3
    max_event_duration_s: float = 50.0

    # --- Event Correlation ---
    # Plausible vehicle speed range (km/h) for computing search windows
    speed_min_kmh: float = 50.0
    speed_max_kmh: float = 150.0
    # Maximum allowed shape dissimilarity (normalized cross-correlation < this => reject)
    min_shape_similarity: float = 0.1
    # Maximum allowed amplitude ratio difference for matching
    max_amplitude_ratio: float = 3.0
    # Minimum amplitude ratio
    min_amplitude_ratio: float = 0.1

    # --- Speed Estimation ---
    # Reject speed estimates outside this range as outliers
    speed_reject_min_kmh: float = 20.0
    speed_reject_max_kmh: float = 200.0

    # --- Predefined Sensor Groups ---
    # Sensor order follows vehicle travel direction
    # IMPORTANT: User must provide actual inter-sensor distances
    sensor_groups: dict = field(default_factory=lambda: {
        'campate1a': {
            'sensors': ['030911FF_x', '030911EF_x', '03091200_x'],
            'distances': None,
            'description': 'Campate 1a: sensors 106→105→104'
        },
        'campate1a_ext': {
            'sensors': ['030911FF_x', '030911EF_x', '03091200_x',
                        '03091155_z', '03091207_x', '03091119_z'],
            'distances': None,
            'description': 'Campate 1a extended: 6 sensors'
        },
        'campate1b': {
            'sensors': ['0309100F_x', '030910F6_x', '0309101E_x'],
            'distances': None,
            'description': 'Campate 1b: sensors 53→52→51'
        },
        'campate2': {
            'sensors': ['030911D2_x', '03091005_x', '0309101F_x'],
            'distances': None,
            'description': 'Campate 2: sensors 99, 100, 101'
        },
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
# EVENT EXTRACTION (simplified threshold-triggered windowing)
# =============================================================================

@dataclass
class VibrationEvent:
    """A detected vibration event on a single sensor."""
    sensor: str
    start_idx: int             # sample index of event start
    end_idx: int               # sample index of event end
    peak_idx: int              # sample index of peak amplitude (within full signal)
    start_time: float          # seconds from window start
    end_time: float            # seconds from window start
    peak_time: float           # seconds from window start
    peak_amplitude: float      # peak absolute amplitude
    energy: float              # integrated |signal| energy over the event
    duration: float            # seconds
    trigger_times: list = field(default_factory=list, repr=False)
    waveform: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def extract_events(signal: np.ndarray, dt: float, threshold: float,
                   window_sec: float = 5.0, pre_trigger_sec: float = 0.5) -> list[dict]:
    """
    Extract event windows from raw signal using threshold trigger.

    Scan |signal|; when it crosses threshold, collect window_sec of data.
    If threshold is crossed again before window ends, extend the window.
    Include pre_trigger_sec before the first trigger.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (DC-removed).
    dt : float
        Sampling period in seconds.
    threshold : float
        Amplitude threshold to trigger an event window.
    window_sec : float
        Duration of event window after trigger (seconds).
    pre_trigger_sec : float
        Time before trigger to include in the window (seconds).

    Returns
    -------
    list[dict]
        Each dict contains: 'signal', 'start_idx', 'end_idx',
        'start_time', 'end_time', 'trigger_times'.
    """
    n = len(signal)
    window_samples = int(window_sec / dt)
    pre_trigger_samples = int(pre_trigger_sec / dt)

    abs_signal = np.abs(signal)
    events = []
    i = 0

    while i < n:
        if abs_signal[i] >= threshold:
            trigger_times = [i * dt]
            start = max(0, i - pre_trigger_samples)
            end = min(n, i + window_samples)

            # Extend window if re-triggered before it closes
            j = i + 1
            while j < end and j < n:
                if abs_signal[j] >= threshold:
                    new_end = min(n, j + window_samples)
                    if new_end > end:
                        trigger_times.append(j * dt)
                        end = new_end
                j += 1

            events.append({
                'signal': signal[start:end].copy(),
                'start_idx': start,
                'end_idx': end,
                'start_time': start * dt,
                'end_time': end * dt,
                'trigger_times': trigger_times,
            })
            i = end  # jump past this event
        else:
            i += 1

    return events


def extract_and_build_events(
    signal: np.ndarray,
    sensor_name: str,
    config: Config,
) -> tuple[list[VibrationEvent], np.ndarray]:
    """
    Run extract_events on a DC-removed signal, then convert the raw event
    dicts into VibrationEvent dataclass instances with computed peak info,
    energy, and duration filtering.

    Also produces a windowed copy of the signal where everything outside
    detected event windows is zeroed out.

    Parameters
    ----------
    signal : np.ndarray
        DC-removed signal (mean-subtracted).
    sensor_name : str
        Sensor identifier for labeling.
    config : Config
        Pipeline configuration.

    Returns
    -------
    tuple[list[VibrationEvent], np.ndarray]
        - Filtered and enriched events, sorted by peak_time.
        - Windowed signal: zeros everywhere except inside event windows.
    """
    dt = config.dt
    min_samples = int(config.min_event_duration_s / dt)
    max_samples = int(config.max_event_duration_s / dt)

    raw_events = extract_events(
        signal, dt,
        threshold=config.event_threshold,
        window_sec=config.event_window_sec,
        pre_trigger_sec=config.event_pre_trigger_sec,
    )

    print(f"    {sensor_name}: {len(raw_events)} raw event windows extracted")

    # Build a zeroed-out copy: keep signal only inside event windows
    windowed_signal = np.zeros_like(signal)
    for ev in raw_events:
        s, e = ev['start_idx'], ev['end_idx']
        windowed_signal[s:e] = signal[s:e]

    active_samples = sum(ev['end_idx'] - ev['start_idx'] for ev in raw_events)
    print(f"    {sensor_name}: windowed signal active ratio: "
          f"{active_samples / len(signal) * 100:.2f}%  "
          f"({active_samples}/{len(signal)} samples)")

    vibration_events = []
    for ev in raw_events:
        duration_samples = ev['end_idx'] - ev['start_idx']

        # Duration filtering
        if duration_samples < min_samples:
            continue
        if duration_samples > max_samples:
            continue

        seg = ev['signal']  # waveform within the event window
        s_idx = ev['start_idx']
        e_idx = ev['end_idx']

        # Peak: index of max |signal| within the event, in full-signal coordinates
        peak_local = np.argmax(np.abs(seg))
        peak_idx = s_idx + peak_local
        peak_amplitude = np.abs(seg[peak_local])

        # Energy: sum of |signal| over the event
        energy = np.sum(np.abs(seg)) * dt

        vibration_events.append(VibrationEvent(
            sensor=sensor_name,
            start_idx=s_idx,
            end_idx=e_idx,
            peak_idx=peak_idx,
            start_time=s_idx * dt,
            end_time=e_idx * dt,
            peak_time=peak_idx * dt,
            peak_amplitude=peak_amplitude,
            energy=energy,
            duration=(e_idx - s_idx) * dt,
            trigger_times=ev['trigger_times'],
            waveform=seg,
        ))

    vibration_events.sort(key=lambda v: v.peak_time)
    print(f"    {sensor_name}: {len(vibration_events)} events after duration filter "
          f"[{config.min_event_duration_s}s, {config.max_event_duration_s}s]")
    return vibration_events, windowed_signal


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
    Compute time delay between two signals using normalized cross-correlation.

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
    s1 = sig1
    s2 = sig2

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
    windowed_signals: dict[str, np.ndarray],
    config: Config
) -> list[VehiclePass]:
    """
    Match vibration events across consecutive sensors to identify vehicle passages.

    For each event on the first sensor, search for matching events on
    subsequent sensors within time windows derived from the plausible
    speed range and known inter-sensor distances. Cross-correlation on
    the windowed signal (zeroed outside events) refines the match.

    Parameters
    ----------
    all_events : dict
        {sensor_name: [VibrationEvent, ...]} for each sensor.
    sensor_order : list[str]
        Sensors in order of vehicle travel direction.
    distances : list[float]
        Cumulative distances from first sensor (meters).
    windowed_signals : dict
        {sensor_name: np.ndarray} signals zeroed outside event windows.
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

            # Distance from previous sensor
            d = distances[i] - distances[i - 1]  # meters
            if d <= 0:
                print(f"  Warning: non-positive distance between {sensor_order[i-1]} "
                      f"and {next_sensor}: {d}m")
                valid_chain = False
                break

            # Time window: [d/v_max, d/v_min] from the previous event's peak
            prev_event = matched_chain[-1]
            t_min_delay = d / v_max  # fastest → smallest delay
            t_max_delay = d / v_min  # slowest → largest delay

            expected_t_min = prev_event.peak_time + t_min_delay
            expected_t_max = prev_event.peak_time + t_max_delay

            # Find candidate events within the time window
            candidates = []
            count=0
            for j, ev in enumerate(next_events):
                if j in used_events[next_sensor]:
                    continue
                if expected_t_min <= ev.peak_time <= expected_t_max:
                    amp_ratio = abs(ev.peak_amplitude) / (abs(anchor_event.peak_amplitude) + 1e-10)
                    if config.min_amplitude_ratio <= amp_ratio <= config.max_amplitude_ratio:
                        count = count+1
                        print('count:', count)
                        candidates.append((j, ev))

            if not candidates:
                valid_chain = False
                break

            # Refine with cross-correlation on the DC-removed signal
            prev_ev = matched_chain[-1]
            seg_half = int(max(prev_ev.duration, 0.5) / dt)
            seg1_start = max(0, prev_ev.peak_idx - seg_half)
            seg1_end = min(len(windowed_signals[prev_ev.sensor]),
                          prev_ev.peak_idx + seg_half)
            seg1 = windowed_signals[prev_ev.sensor][seg1_start:seg1_end]

            max_lag = int(t_max_delay / dt) + seg_half

            best_match = None
            best_corr = -1.0
            best_delay = 0.0

            for j, cand_ev in candidates:
                seg2_start = max(0, cand_ev.peak_idx - seg_half)
                seg2_end = min(len(windowed_signals[next_sensor]),
                              cand_ev.peak_idx + seg_half)
                seg2 = windowed_signals[next_sensor][seg2_start:seg2_end]

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
                #print(vp.events)
                vehicle_passes.append(vp)

    vehicle_passes.sort(key=lambda vp: vp.events[0].peak_time)
    return vehicle_passes


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_detection_results(
    windowed_signals: dict[str, np.ndarray],
    all_events: dict[str, list[VibrationEvent]],
    sensor_order: list[str],
    config: Config,
    save_path: str = 'graphs/event_detection.png'
):
    """
    Plot DC-removed signals with detected event windows highlighted.
    """
    n_sensors = len(sensor_order)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3.5 * n_sensors),
                             sharex=True)
    if n_sensors == 1:
        axes = [axes]

    time_s = np.arange(len(windowed_signals[sensor_order[0]])) * config.dt
    colors = plt.cm.Set1(np.linspace(0, 1, max(10, n_sensors)))

    for i, sensor in enumerate(sensor_order):
        ax = axes[i]
        sig = windowed_signals[sensor]
        events = all_events.get(sensor, [])

        # Plot signal
        ax.plot(time_s, sig, color=colors[i], alpha=0.6, linewidth=0.5,
                label=f'{sensor}')

        # Threshold lines
        thr = config.event_threshold
        ax.axhline(thr, color='red', linestyle='--', alpha=0.5,
                    linewidth=0.8, label=f'Threshold ({thr})')
        ax.axhline(-thr, color='red', linestyle='--', alpha=0.5,
                    linewidth=0.8)

        # Highlight detected event windows
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
    Plot matched vehicle passages:
      - Top: Time-distance diagram with vehicle tracks
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
    windowed_signals: dict[str, np.ndarray],
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
            end = min(len(windowed_signals[ev.sensor]), ev.peak_idx + half_win)
            wave = windowed_signals[ev.sensor][start:end]

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
        print("  Try: lower event_threshold, increase event_window_sec, or longer time window.")

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

    Pipeline flow:
        1. Load data → time-windowed DataFrame
        2. DC removal (mean subtraction) per sensor
        3. Event extraction via threshold-triggered windowing
        4. Cross-sensor event correlation → speed estimates
        5. Visualization & summary

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
    print(f"  File:            {filepath}")
    print(f"  Window:          {start_time} + {duration_mins} min")
    print(f"  Sensors:         {sensor_order}")
    print(f"  Distances:       {distances} m")
    print(f"  Speed range:     {config.speed_min_kmh}–{config.speed_max_kmh} km/h")
    print(f"  Event threshold: {config.event_threshold}")
    print(f"  Event window:    {config.event_window_sec}s  "
          f"pre-trigger: {config.event_pre_trigger_sec}s")
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

    # Step 2: DC removal (mean subtraction) per sensor
    print("\nStep 2: DC removal (mean subtraction)...")
    dc_removed_signals = {}
    for sensor in sensor_order:
        raw = df[sensor].values.astype(np.float64)
        centered = raw - np.nanmean(raw)
        dc_removed_signals[sensor] = centered
        print(f"  {sensor}: raw [{np.nanmin(raw):.6f}, {np.nanmax(raw):.6f}] "
              f"→ centered [{centered.min():.6f}, {centered.max():.6f}]")

    # Step 3: Extract events per sensor
    # Also produces windowed signals (zeroed outside event windows)
    print("\nStep 3: Extracting events (threshold-triggered windowing)...")
    all_events = {}
    windowed_signals = {}
    for sensor in sensor_order:
        events, windowed = extract_and_build_events(
            dc_removed_signals[sensor],
            sensor,
            config,
        )
        all_events[sensor] = events
        windowed_signals[sensor] = windowed
        print(f"  {sensor}: {len(events)} events")

    # Step 4: Correlate events & estimate speed
    # Uses windowed signals (zeroed outside events) for cross-correlation
    print("\nStep 4: Correlating events across sensors...")
    vehicle_passes = correlate_events_across_sensors(
        all_events, sensor_order, distances,
        windowed_signals, config
    )

    # Step 5: Visualize
    print("\nStep 5: Generating visualizations...")
    plot_detection_results(
        windowed_signals, all_events,
        sensor_order, config,
        save_path=os.path.join(output_dir, 'event_detection.png')
    )
    plot_correlation_results(
        vehicle_passes, sensor_order, distances, config,
        save_path=os.path.join(output_dir, 'speed_estimation.png')
    )
    plot_matched_waveforms(
        vehicle_passes, windowed_signals, config,
        save_path=os.path.join(output_dir, 'matched_waveforms.png')
    )

    # Step 6: Summary
    print_summary(vehicle_passes, all_events, sensor_order, distances)

    diagnostics = {
        'df': df,
        'dc_removed_signals': dc_removed_signals,
        'windowed_signals': windowed_signals,
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

    # Distances (always required)
    parser.add_argument('--distances', nargs='+', type=float, required=True,
                        help='Cumulative distance (m) of each sensor from first sensor. '
                             'Must start with 0. E.g.: 0 25 50')

    # Event extraction parameters
    parser.add_argument('--event_threshold', type=float, default=0.001,
                        help='Amplitude threshold to trigger event window (default: 0.001)')
    parser.add_argument('--event_window_sec', type=float, default=4.0,
                        help='Event window duration after trigger in seconds (default: 5.0)')
    parser.add_argument('--event_pre_trigger_sec', type=float, default=1.0,
                        help='Pre-trigger time to include in seconds (default: 0.5)')

    # Correlation / speed parameters
    parser.add_argument('--speed_min', type=float, default=60.0,
                        help='Min plausible speed km/h (default: 60)')
    parser.add_argument('--speed_max', type=float, default=150.0,
                        help='Max plausible speed km/h (default: 130)')
    parser.add_argument('--min_event_duration', type=float, default=0.3,
                        help='Min event duration seconds (default: 0.3)')
    parser.add_argument('--max_event_duration', type=float, default=100.0,
                        help='Max event duration seconds (default: 100.0)')
    parser.add_argument('--b', type=float, default=0.1,
                        help='Min cross-correlation score for matching (default: 0.1)')
    parser.add_argument('--output_dir', default='graphs/approach_2_cross_corr_wt_confidence',
                        help='Output directory for plots')

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()

    # Event extraction parameters
    config.event_threshold      = args.event_threshold
    config.event_window_sec     = args.event_window_sec
    config.event_pre_trigger_sec = args.event_pre_trigger_sec

    # Duration filters
    config.min_event_duration_s = args.min_event_duration
    config.max_event_duration_s = args.max_event_duration

    # Correlation parameters
    config.speed_min_kmh        = args.speed_min
    config.speed_max_kmh        = args.speed_max
    config.min_shape_similarity = args.b

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

# ex: python3 vehicle_speed_estimation.py --path /Users/thomas/Data/Data_sensors/20250307/csv_acc/M001_2025-03-07_01-00-00_gg-112_int-2_th.csv --start_time '2025/03/07 01:05:00' --duration_mins 4 --sensors 030911FF_x 030911EF_x 03091155_z --b 0.1 --distances 0 32.36 83.91