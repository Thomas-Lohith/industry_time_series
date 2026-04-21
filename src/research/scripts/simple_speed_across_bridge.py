#!/usr/bin/env python3
"""
Simple Vehicle Speed Estimation — Multi-Sensor Trace
=====================================================

Minimal 3-stage approach:
  Stage 1: Threshold-triggered event extraction from first sensor
  Stage 2: Trace each event across ALL downstream sensors by sliding
           the template over each sensor's full signal (raw + envelope)
  Stage 3: Speed = distance / lag for each pair, plus overall estimate

Each sensor can be at a DIFFERENT distance from the reference sensor.
The event from sensor[0] is traced to sensor[1], sensor[2], ..., sensor[N-1].

No bandpass filters. Preprocessing = DC removal (mean subtraction) only.

Usage:
    # 3 sensors at positions 0m, 25m, 50m
    python simple_speed.py \\
        --path data.csv \\
        --start_time '2025/03/07 01:05:00' \\
        --duration_mins 5 \\
        --sensors 030911FF_x 030911EF_x 03091200_x \\
        --distances 0 25 50 \\
        --threshold 0.05

    # 4 sensors at non-uniform spacing
    python simple_speed.py \\
        --path data.csv \\
        --start_time '2025/03/07 01:05:00' \\
        --duration_mins 5 \\
        --sensors S1 S2 S3 S4 \\
        --distances 0 18.5 40.0 72.3 \\
        --threshold 0.05

    # Use bridge model distances from campata_2 left side:
    # e.g. 0, 10.67, 21.33, 32.0
    python simple_speed.py \\
        --path data.parquet \\
        --start_time '2025/03/07 01:05:00' \\
        --duration_mins 3 \\
        --sensors 030911D2_x 03091005_x 0309101F_x \\
        --distances 0 10.67 21.33 \\
        --threshold 0.04
"""

import argparse
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# STAGE 1: THRESHOLD-TRIGGERED EVENT EXTRACTION
# =============================================================================

def extract_events(signal, dt, threshold, window_sec=5.0, pre_trigger_sec=0.5):
    """
    Extract event windows from raw signal using threshold trigger.

    Scan |signal|, when it crosses threshold -> collect window_sec of data.
    If threshold is crossed again before window ends -> extend.
    Include pre_trigger_sec before first trigger.
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

            # Extend window if re-triggered
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
                'trigger_times': trigger_times
            })
            i = end
        else:
            i += 1

    return events


# =============================================================================
# STAGE 2: SLIDE EVENT TEMPLATE OVER FULL SIGNAL
# =============================================================================

def normalize(sig):
    """Zero mean, unit std — removes sensor sensitivity differences."""
    sig = sig - np.mean(sig)
    s = np.std(sig)
    if s > 1e-10:
        sig = sig / s
    return sig


def compute_envelope(sig, window=30):
    """Sliding RMS envelope."""
    sq = sig ** 2
    cs = np.cumsum(np.insert(sq, 0, 0))
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    return rms


def slide_event_over_signal(event_signal, full_signal_b, event_start_idx,
                            dt, max_lag_sec=5.0, use_envelope=False):
    """
    Slide event template from sensor A over the full signal of sensor B.

    Preserves absolute time reference:
        - Event on A starts at event_start_idx
        - Search B's signal from that point forward
        - lag = (best_match_position - event_start_idx) * dt
    """
    max_lag_samples = int(max_lag_sec / dt)

    if use_envelope:
        template = compute_envelope(normalize(event_signal))
        target = compute_envelope(normalize(full_signal_b))
    else:
        template = normalize(event_signal)
        target = normalize(full_signal_b)

    template_len = len(template)
    target_len = len(target)

    if template_len < 10 or target_len < template_len:
        return 0.0, 0.0, np.array([]), np.array([])

    search_start = event_start_idx
    search_end = min(target_len - template_len, event_start_idx + max_lag_samples)

    if search_end <= search_start:
        return 0.0, 0.0, np.array([]), np.array([])

    template_norm = np.linalg.norm(template)
    if template_norm < 1e-10:
        return 0.0, 0.0, np.array([]), np.array([])

    lags = []
    corrs = []

    for pos in range(search_start, search_end):
        segment = target[pos:pos + template_len]
        if len(segment) < template_len:
            break

        seg_norm = np.linalg.norm(segment)
        if seg_norm < 1e-10:
            corrs.append(0.0)
        else:
            corr = np.dot(template, segment) / (template_norm * seg_norm)
            corrs.append(corr)

        lag_samples = pos - event_start_idx
        lags.append(lag_samples * dt)

    if not corrs:
        return 0.0, 0.0, np.array([]), np.array([])

    lags = np.array(lags)
    corrs = np.array(corrs)

    best_idx = np.argmax(corrs)
    best_lag_sec = lags[best_idx]
    best_corr = corrs[best_idx]

    return best_lag_sec, float(best_corr), lags, corrs


# =============================================================================
# STAGE 3: SPEED FROM LAG
# =============================================================================

def estimate_speed(lag_sec, distance_m):
    """speed = distance / time, returns km/h"""
    if lag_sec <= 0:
        return float('inf')
    return (distance_m / lag_sec) * 3.6


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_stage1(raw_signals, events_first_sensor, sensor_names, distances,
                dt, threshold, save_path):
    """
    Plot all sensor signals with events from the FIRST sensor highlighted.
    Shows sensor name + distance on y-axis labels.
    """
    n_sensors = len(sensor_names)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i, sensor in enumerate(sensor_names):
        ax = axes[i]
        raw = raw_signals[sensor]
        t = np.arange(len(raw)) * dt

        ax.plot(t, raw, color='steelblue', linewidth=0.4, alpha=0.7, label='Signal')
        ax.axhline(+threshold, color='red', linestyle='--', linewidth=0.8,
                    alpha=0.6, label=f'threshold +/-{threshold}')
        ax.axhline(-threshold, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

        # Highlight events (only from first sensor)
        if i == 0:
            for k, ev in enumerate(events_first_sensor):
                ax.axvspan(ev['start_time'], ev['end_time'],
                           alpha=0.15, color='orange',
                           label='Detected event' if k == 0 else None)
                for tt in ev['trigger_times']:
                    ax.axvline(tt, color='green', linewidth=0.5, alpha=0.4)

        ax.set_ylabel(f'{sensor}\n({distances[i]:.1f}m)', fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        n_ev = len(events_first_sensor) if i == 0 else '-'
        ax.set_title(f'{sensor} @ {distances[i]:.1f}m  '
                     f'(events: {n_ev})', fontsize=10)

    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    fig.suptitle('Stage 1: All Sensors + Events Detected on First Sensor',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_event_trace(trace, sensors, distances, event_idx, dt, save_path):
    """
    Plot the trace of a single event across all sensors:
      Top:    Time-distance diagram (lag vs distance)
      Bottom: Correlation scores per sensor
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    # Unpack trace results
    matched_sensors = [sensors[0]]   # first sensor is the anchor
    matched_dists = [distances[0]]
    matched_lags = [0.0]
    matched_corrs = [1.0]            # perfect self-correlation at lag=0

    for r in trace:
        matched_sensors.append(r['sensor'])
        matched_dists.append(r['distance'])
        matched_lags.append(r['best_lag'])
        matched_corrs.append(r['best_score'])

    # --- Top: Time-Distance diagram ---
    ax = axes[0]
    ax.plot(matched_lags, matched_dists, 'o-', color='royalblue',
            linewidth=2, markersize=10, zorder=5)

    # Label each point with sensor name
    for lag, dist, name in zip(matched_lags, matched_dists, matched_sensors):
        ax.annotate(f'{name}\n{lag:.3f}s', xy=(lag, dist),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=7, color='darkblue')

    # Draw speed line (if at least 2 points with valid lag)
    valid = [(l, d) for l, d in zip(matched_lags, matched_dists) if l > 0]
    if valid:
        # Linear fit: distance = speed * lag
        lags_v = np.array([0.0] + [v[0] for v in valid])
        dists_v = np.array([0.0] + [v[1] for v in valid])
        if len(lags_v) >= 2 and lags_v[-1] > 0:
            # Fit line through origin: slope = speed_m_s
            speed_ms = np.polyfit(lags_v, dists_v, 1)[0]
            speed_kmh = speed_ms * 3.6
            fit_lags = np.linspace(0, max(lags_v) * 1.1, 50)
            fit_dists = speed_ms * fit_lags
            ax.plot(fit_lags, fit_dists, '--', color='red', linewidth=1.5,
                    alpha=0.7, label=f'Fit: {speed_kmh:.1f} km/h')
            ax.legend(fontsize=10)

    ax.set_xlabel('Lag from first sensor (seconds)', fontsize=11)
    ax.set_ylabel('Distance from first sensor (m)', fontsize=11)
    ax.set_title(f'Event {event_idx+1}: Trace Across Sensors',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Bottom: Correlation scores ---
    ax2 = axes[1]
    sensor_labels = [f'{s}\n{d:.1f}m' for s, d in zip(matched_sensors, matched_dists)]
    colors = ['green' if c > 0.7 else 'orange' if c > 0.4 else 'red'
              for c in matched_corrs]
    bars = ax2.bar(range(len(matched_corrs)), matched_corrs, color=colors,
                   edgecolor='white', alpha=0.8)
    ax2.set_xticks(range(len(sensor_labels)))
    ax2.set_xticklabels(sensor_labels, fontsize=7)
    ax2.set_ylabel('Correlation', fontsize=11)
    ax2.set_title('Match Quality Per Sensor', fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.7, color='green', linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.axhline(0.4, color='orange', linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add correlation value on each bar
    for bar, c in zip(bars, matched_corrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{c:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stage2(event_sig, full_sig_b, sensor_a, sensor_b, dist_b,
                event_start_time, dt,
                lags_raw, corr_raw, lag_raw, score_raw,
                lags_env, corr_env, lag_env, score_env,
                save_path):
    """Plot correlation curves for one sensor pair."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Top: signals overlay
    ax = axes[0]
    t_b = np.arange(len(full_sig_b)) * dt
    ax.plot(t_b, full_sig_b, color='gray', linewidth=0.3, alpha=0.5,
            label=f'{sensor_b} full signal')
    t_ev = event_start_time + np.arange(len(event_sig)) * dt
    ax.plot(t_ev, event_sig, color='blue', linewidth=1.0, alpha=0.9,
            label=f'{sensor_a} event (template)')

    if lag_raw > 0:
        ax.axvline(event_start_time + lag_raw, color='red', linewidth=1.5,
                    linestyle='--', label=f'Best raw match')
    if lag_env > 0 and abs(lag_env - lag_raw) > 0.01:
        ax.axvline(event_start_time + lag_env, color='green', linewidth=1.5,
                    linestyle='--', label=f'Best envelope match')

    ax.set_title(f'Template slid over {sensor_b} ({dist_b:.1f}m)', fontsize=11)
    ax.set_xlabel('Time (seconds)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: raw correlation
    ax = axes[1]
    if len(lags_raw) > 0:
        ax.plot(lags_raw, corr_raw, color='blue', linewidth=0.8)
        ax.axvline(lag_raw, color='red', linestyle='--', linewidth=1.5,
                    label=f'Peak: lag={lag_raw:.3f}s, corr={score_raw:.3f}')
        ax.legend(fontsize=9)
    ax.set_title('Sliding Correlation (raw)', fontsize=11)
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lag (seconds)')
    ax.grid(True, alpha=0.3)

    # Bottom: envelope correlation
    ax = axes[2]
    if len(lags_env) > 0:
        ax.plot(lags_env, corr_env, color='darkgreen', linewidth=0.8)
        ax.axvline(lag_env, color='red', linestyle='--', linewidth=1.5,
                    label=f'Peak: lag={lag_env:.3f}s, corr={score_env:.3f}')
        ax.legend(fontsize=9)
    ax.set_title('Sliding Correlation (envelope)', fontsize=11)
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lag (seconds)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stage3_aligned(event_sig, full_sig_b, sensor_a, sensor_b,
                        event_start_idx, best_lag_sec, dt, save_path):
    """Show event from A overlaid on B at the matched position."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    norm_a = normalize(event_sig)
    t_a = np.arange(len(norm_a)) * dt

    lag_samples = int(best_lag_sec / dt)
    match_start = event_start_idx + lag_samples
    match_end = match_start + len(event_sig)

    if match_end <= len(full_sig_b):
        matched_b = full_sig_b[match_start:match_end]
        norm_b = normalize(matched_b)
        ax.plot(t_a, norm_a, color='blue', linewidth=1.0, alpha=0.8,
                label=f'{sensor_a} event')
        ax.plot(t_a, norm_b, color='red', linewidth=1.0, alpha=0.8,
                label=f'{sensor_b} matched segment')
        ax.set_title(f'Aligned Comparison (lag = {best_lag_sec:.3f}s)',
                     fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Matched segment extends beyond signal',
                ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Time within event (seconds)')
    ax.set_ylabel('Normalized amplitude')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath, sensor_columns, start_time, duration_mins):
    """Load from CSV or Parquet, filter to time window."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.parquet':
        df_pl = pl.scan_parquet(filepath)
        available = list(df_pl.collect_schema().keys())
    elif ext == '.csv':
        df_pl = pl.read_csv(filepath, separator =';')
        available = df_pl.columns
    else:
        raise ValueError(f"Unsupported: {ext}")

    time_col = None
    for c in ['time', 'Time', 'timestamp', 'Timestamp']:
        if c in available:
            time_col = c
            break
    if not time_col:
        raise ValueError(f"No time column in: {available[:10]}")

    missing = [s for s in sensor_columns if s not in available]
    if missing:
        raise ValueError(f"Sensors not found: {missing}")

    cols = [time_col] + sensor_columns
    if ext == '.parquet':
        df = df_pl.select(cols).collect().to_pandas()
    else:
        df = df_pl.select(cols).to_pandas()

    df['time'] = pd.to_datetime(df[time_col], format='%Y/%m/%d %H:%M:%S:%f',
                                 errors='coerce', exact=False)
    if time_col != 'time':
        df = df.drop(columns=[time_col])

    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

    t_start = pd.to_datetime(start_time)
    t_end = t_start + pd.Timedelta(minutes=duration_mins)
    df = df[(df['time'] >= t_start) & (df['time'] <= t_end)].reset_index(drop=True)

    print(f"Loaded {len(df):,} samples")
    print(f"  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(filepath, start_time, duration_mins, sensors, distances, threshold,
        window_sec=5.0, pre_trigger_sec=0.5, max_lag_sec=5.0, output_dir='graphs/across_bridge'):
    """
    Run the 3-stage pipeline.

    Parameters
    ----------
    sensors : list[str]
        Sensor IDs in vehicle travel order.
    distances : list[float]
        Cumulative distance of each sensor from the first one (meters).
        Must start with 0. E.g. [0, 18.5, 40.0, 72.3]
    """
    dt = 0.01  # 100 Hz

    # --- Validate ---
    assert len(sensors) == len(distances), \
        f"sensors ({len(sensors)}) and distances ({len(distances)}) must match"
    assert distances[0] == 0, \
        f"First distance must be 0 (reference sensor), got {distances[0]}"
    for i in range(1, len(distances)):
        assert distances[i] > distances[i-1], \
            f"Distances must increase: distances[{i}]={distances[i]} <= distances[{i-1}]={distances[i-1]}"

    print("=" * 60)
    print("  SIMPLE VEHICLE SPEED ESTIMATION")
    print("  Multi-Sensor Trace")
    print("=" * 60)
    print(f"\n  Sensor layout:")
    for s, d in zip(sensors, distances):
        print(f"    {s:20s}  @ {d:7.1f}m")
    print(f"\n  Gaps between consecutive sensors:")
    for i in range(1, len(sensors)):
        gap = distances[i] - distances[i-1]
        print(f"    {sensors[i-1]} -> {sensors[i]}: {gap:.1f}m")
    print(f"\n  Total span:   {distances[-1] - distances[0]:.1f}m")
    print(f"  Threshold:    {threshold}")
    print(f"  Window:       {window_sec}s + {pre_trigger_sec}s pre-trigger")
    print(f"  Max lag:      {max_lag_sec}s")
    print()

    # --- Load ---
    df = load_data(filepath, sensors, start_time, duration_mins)

    # --- Preprocess all sensors (DC removal only) ---
    raw_signals = {}
    for sensor in sensors:
        raw = df[sensor].values.astype(np.float64)
        raw = raw - np.mean(raw)
        raw_signals[sensor] = raw

    # =====================================================================
    # STAGE 1: Extract events from FIRST sensor only
    # =====================================================================
    print("\n--- STAGE 1: Event Extraction (first sensor) ---")
    first_sensor = sensors[0]
    events = extract_events(
        raw_signals[first_sensor], dt, threshold, window_sec, pre_trigger_sec
    )
    print(f"  {first_sensor}: {len(events)} event(s)")
    for k, ev in enumerate(events):
        dur = ev['end_time'] - ev['start_time']
        print(f"    Event {k+1}: {ev['start_time']:.2f}s - {ev['end_time']:.2f}s "
              f"({dur:.2f}s, {len(ev['trigger_times'])} triggers)")

    if not events:
        print("\n  No events detected. Try lowering --threshold.")
        return []

    plot_stage1(raw_signals, events, sensors, distances, dt, threshold,
                os.path.join(output_dir, 'stage1_events.png'))

    # =====================================================================
    # STAGE 2 & 3: Trace each event across all downstream sensors
    # =====================================================================
    print("\n--- STAGE 2 & 3: Trace Events Across Sensors ---")

    all_traces = []  # one trace per event

    for ev_idx, ev in enumerate(events):
        print(f"\n  Event {ev_idx+1} (from {first_sensor}, "
              f"{ev['start_time']:.2f}s - {ev['end_time']:.2f}s)")
        print(f"  {'─' * 50}")

        trace = []  # results for this event across downstream sensors

        for s_idx in range(1, len(sensors)):
            sensor_b = sensors[s_idx]
            dist_from_first = distances[s_idx]  # cumulative from sensor[0]
            full_b = raw_signals[sensor_b]

            # Slide event template over this sensor's full signal
            lag_raw, score_raw, lags_r, corr_r = slide_event_over_signal(
                ev['signal'], full_b, ev['start_idx'],
                dt, max_lag_sec, use_envelope=False
            )
            lag_env, score_env, lags_e, corr_e = slide_event_over_signal(
                ev['signal'], full_b, ev['start_idx'],
                dt, max_lag_sec, use_envelope=True
            )

            # Pick better method
            if score_env > score_raw:
                best_lag = lag_env
                best_score = score_env
                method = "envelope"
            else:
                best_lag = lag_raw
                best_score = score_raw
                method = "raw"

            best_speed = estimate_speed(best_lag, dist_from_first)

            # Also compute pair-wise speed (gap to previous sensor)
            if s_idx == 1:
                prev_lag = 0.0
            else:
                prev_lag = trace[-1]['best_lag']
            pair_gap = distances[s_idx] - distances[s_idx - 1]
            pair_delay = best_lag - prev_lag
            pair_speed = estimate_speed(pair_delay, pair_gap)

            result = {
                'sensor': sensor_b,
                'sensor_idx': s_idx,
                'distance': dist_from_first,
                'pair_gap': pair_gap,
                'lag_raw': lag_raw, 'score_raw': score_raw,
                'lag_env': lag_env, 'score_env': score_env,
                'best_lag': best_lag, 'best_score': best_score,
                'best_method': method, 'best_speed': best_speed,
                'pair_delay': pair_delay, 'pair_speed': pair_speed,
                'lags_r': lags_r, 'corr_r': corr_r,
                'lags_e': lags_e, 'corr_e': corr_e,
            }
            trace.append(result)

            # Print per-sensor results
            print(f"\n    -> {sensor_b} ({dist_from_first:.1f}m from {first_sensor})")
            print(f"       RAW:       corr={score_raw:.4f}  lag={lag_raw:.4f}s")
            print(f"       ENVELOPE:  corr={score_env:.4f}  lag={lag_env:.4f}s")
            print(f"       BEST:      {method} | lag={best_lag:.3f}s | corr={best_score:.3f}")
            print(f"       Cumulative speed (0 -> {dist_from_first:.1f}m): ", end="")
            if 5 < best_speed < 300:
                print(f"{best_speed:.1f} km/h")
            else:
                print(f"{best_speed:.1f} km/h (suspect)")
            print(f"       Pair speed ({sensors[s_idx-1]} -> {sensor_b}, "
                  f"{pair_gap:.1f}m, dt={pair_delay:.3f}s): ", end="")
            if 5 < pair_speed < 300:
                print(f"{pair_speed:.1f} km/h")
            else:
                print(f"{pair_speed:.1f} km/h (suspect)")

            # Per-sensor correlation plots
            tag = f"ev{ev_idx+1}_s{s_idx}"
            plot_stage2(
                ev['signal'], full_b, first_sensor, sensor_b, dist_from_first,
                ev['start_time'], dt,
                lags_r, corr_r, lag_raw, score_raw,
                lags_e, corr_e, lag_env, score_env,
                os.path.join(output_dir, f'stage2_{tag}.png')
            )
            plot_stage3_aligned(
                ev['signal'], full_b, first_sensor, sensor_b,
                ev['start_idx'], best_lag, dt,
                os.path.join(output_dir, f'stage3_{tag}.png')
            )

        # --- Event trace summary ---
        all_traces.append(trace)

        # Trace plot: time-distance diagram for this event
        plot_event_trace(trace, sensors, distances, ev_idx, dt,
                         os.path.join(output_dir, f'trace_ev{ev_idx+1}.png'))

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for ev_idx, trace in enumerate(all_traces):
        ev = events[ev_idx]
        print(f"\n  Event {ev_idx+1} @ {ev['start_time']:.2f}s on {first_sensor}:")

        # Cumulative speed (first -> last sensor using total distance and total lag)
        if trace:
            last = trace[-1]
            total_dist = last['distance']
            total_lag = last['best_lag']
            overall_speed = estimate_speed(total_lag, total_dist)
            avg_corr = np.mean([r['best_score'] for r in trace])

            print(f"    Overall: {total_dist:.1f}m in {total_lag:.3f}s "
                  f"= {overall_speed:.1f} km/h (avg corr: {avg_corr:.3f})")

        # Per-pair breakdown
        print(f"    {'Pair':<35s}  {'Gap':>6s}  {'Delay':>7s}  "
              f"{'Speed':>9s}  {'Corr':>6s}  {'Method'}")
        print(f"    {'─'*35}  {'─'*6}  {'─'*7}  {'─'*9}  {'─'*6}  {'─'*8}")

        for r in trace:
            prev_sensor = sensors[r['sensor_idx'] - 1]
            pair_label = f"{prev_sensor} -> {r['sensor']}"
            speed_str = f"{r['pair_speed']:.1f}" if 5 < r['pair_speed'] < 300 else f"{r['pair_speed']:.0f}?"
            print(f"    {pair_label:<35s}  {r['pair_gap']:6.1f}m  "
                  f"{r['pair_delay']:7.3f}s  {speed_str:>8s}  "
                  f"{r['best_score']:6.3f}  {r['best_method']}")

    print("\n" + "=" * 60)
    return all_traces


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Simple vehicle speed estimation — multi-sensor trace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3 sensors at 0m, 25m, 50m:
  python simple_speed.py --path data.csv \\
      --start_time '2025/03/07 01:05:00' --duration_mins 5 \\
      --sensors 030911FF_x 030911EF_x 03091200_x \\
      --distances 0 25 50 --threshold 0.05

  # 4 sensors, non-uniform spacing:
  python simple_speed.py --path data.parquet \\
      --start_time '2025/03/07 01:05:00' --duration_mins 3 \\
      --sensors S1 S2 S3 S4 \\
      --distances 0 18.5 40.0 72.3 --threshold 0.04
        """)
    parser.add_argument('--path', required=True, help='CSV or Parquet file')
    parser.add_argument('--start_time', required=True,
                        help="e.g. '2025/03/07 01:05:00'")
    parser.add_argument('--duration_mins', type=float, required=True)
    parser.add_argument('--sensors', nargs='+', required=True,
                        help='Sensor IDs in vehicle travel order')
    parser.add_argument('--distances', nargs='+', type=float, required=True,
                        help='Cumulative distance (m) of each sensor from first. '
                             'Must start with 0. E.g.: 0 18.5 40.0 72.3')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Amplitude threshold for event trigger')
    parser.add_argument('--window_sec', type=float, default=20.0,
                        help='Collection window after trigger (default: 5.0)')
    parser.add_argument('--pre_trigger_sec', type=float, default=5.0,
                        help='Pre-trigger seconds (default: 0.5)')
    parser.add_argument('--max_lag', type=float, default=50.0,
                        help='Max lag to search in seconds (default: 5.0)')
    parser.add_argument('--output_dir', default='graphs/across_bridge',
                        help='Output directory for plots (default: graphs)')
    args = parser.parse_args()

    # Validate distances count matches sensors
    if len(args.distances) != len(args.sensors):
        parser.error(f"Got {len(args.sensors)} sensors but {len(args.distances)} distances. "
                     f"Must be one distance per sensor.")

    run(filepath=args.path, start_time=args.start_time,
        duration_mins=args.duration_mins, sensors=args.sensors,
        distances=args.distances, threshold=args.threshold,
        window_sec=args.window_sec, pre_trigger_sec=args.pre_trigger_sec,
        max_lag_sec=args.max_lag, output_dir=args.output_dir)


if __name__ == '__main__':
    main()