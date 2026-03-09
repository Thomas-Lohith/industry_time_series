#!/usr/bin/env python3
"""
Simple Vehicle Speed Estimation — Threshold + Slide-Correlate
=============================================================

Minimal approach:
  Stage 1: Threshold-triggered event extraction from raw signal
  Stage 2: Take event from sensor A, slide it over sensor B's full signal
           to find where the pattern appears → gives the true time delay
  Stage 3: Speed = distance / lag

No bandpass filters. Preprocessing = DC removal (mean subtraction) only.

Usage:
    python simple_speed.py \
        --path data.csv \
        --start_time '2025/03/07 01:05:00' \
        --duration_mins 5 \
        --sensors 030911FF_x 030911EF_x 03091200_x \
        --distance 25.0 \
        --threshold 0.05
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

def extract_events(signal, dt, threshold, window_sec, pre_trigger_sec=0.5):
    """
    Extract event windows from raw signal using threshold trigger.

    How it works:
        - Scan through |signal|
        - When |signal| crosses threshold -> start collecting
        - Collect 'window_sec' seconds of data
        - If threshold is crossed again before window ends -> extend window
        - Include 'pre_trigger_sec' of data before the first trigger

    Parameters
    ----------
    signal : np.ndarray
        Raw acceleration signal (DC removed).
    dt : float
        Time step (0.01s for 100 Hz).
    threshold : float
        Amplitude threshold for triggering.
    window_sec : float
        Base collection window after trigger (seconds).
    pre_trigger_sec : float
        Seconds before trigger to include.

    Returns
    -------
    events : list of dict
        'signal'      : np.ndarray -- extracted segment
        'start_idx'   : int        -- start index in original signal
        'end_idx'     : int        -- end index in original signal
        'start_time'  : float      -- seconds
        'end_time'    : float      -- seconds
        'trigger_times': list[float]
    """
    n = len(signal)
    window_samples = int(window_sec / dt)
    print(window_samples)
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
# STAGE 2: SLIDE EVENT PATTERN OVER FULL SIGNAL
# =============================================================================

def normalize(sig):
    """Zero mean, unit std -- removes sensor sensitivity differences."""
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
    Slide the event extracted from sensor A over the full signal of sensor B.

    This preserves the absolute time reference:
        - We know the event on sensor A starts at 'event_start_idx'
        - We search sensor B's signal AFTER that point (vehicle hasn't arrived yet)
        - The position where correlation peaks = where the pattern appears on B
        - lag = (best_position_on_B - event_position_on_A) * dt

    Parameters
    ----------
    event_signal : np.ndarray
        The extracted event segment from sensor A.
    full_signal_b : np.ndarray
        The complete signal from sensor B.
    event_start_idx : int
        Where this event starts in sensor A's timeline (sample index).
    dt : float
        Time step.
    max_lag_sec : float
        Maximum lag to search ahead (seconds).
    use_envelope : bool
        If True, match envelopes instead of raw waveforms.

    Returns
    -------
    best_lag_sec : float
        Time delay (seconds). Positive = B is later (expected).
    best_corr : float
        Correlation score at best position.
    all_lags : np.ndarray
        All tested lag values.
    all_corrs : np.ndarray
        Correlation at each lag.
    """
    max_lag_samples = int(max_lag_sec / dt)

    # if use_envelope:
    #     template = compute_envelope(normalize(event_signal))
    #     target = compute_envelope(normalize(full_signal_b))
    # else:
    #     template = normalize(event_signal)
    #     target = normalize(full_signal_b)

    ##lets try without normalising the signal
    if use_envelope:
        template = compute_envelope(event_signal)
        target = compute_envelope(full_signal_b)
    else:
        template = event_signal
        target = full_signal_b
    
    print(template)
    print(len(template))

    template_len = len(template)
    target_len = len(target)

    if template_len < 10 or target_len < template_len:
        return 0.0, 0.0, np.array([]), np.array([])

    # Search range: from event_start_idx (lag=0) to event_start_idx + max_lag
    # lag=0 means the pattern appears at the same absolute time on B
    # lag>0 means B sees the pattern later (vehicle moving A->B)
    search_start = event_start_idx
    search_end = min(target_len - template_len, event_start_idx + max_lag_samples)

    if search_end <= search_start:
        return 0.0, 0.0, np.array([]), np.array([])

    # Slide template over target signal
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
    if lag_sec == 0:
        return float('inf')
    return (distance_m / lag_sec) * 3.6


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_stage1(raw_signals, events_per_sensor, sensor_names, dt, threshold, save_path):
    """Plot raw signals with extracted events highlighted."""
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

        for k, ev in enumerate(events_per_sensor[sensor]):
            ax.axvspan(ev['start_time'], ev['end_time'],
                       alpha=0.15, color='orange', label='Event' if k == 0 else None)
            #for tt in ev['trigger_times']:
                #ax.axvline(tt, color='green', linewidth=0.5, alpha=0.4)

        ax.set_ylabel(sensor, fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_title(f'{sensor}: {len(events_per_sensor[sensor])} event(s)', fontsize=10)

    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    fig.suptitle('Stage 1: Threshold Event Extraction', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stage2(event_sig, full_sig_b, sensor_a, sensor_b,
                event_start_time, dt,
                lags_raw, corr_raw, lag_raw, score_raw,
                lags_env, corr_env, lag_env, score_env,
                save_path):
    """Plot correlation curves and the sliding concept."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Top: Show both full signals with event highlighted
    ax = axes[0]
    t_b = np.arange(len(full_sig_b)) * dt
    ax.plot(t_b, full_sig_b, color='gray', linewidth=0.3, alpha=0.5,
            label=f'{sensor_b} full signal')

    t_ev = event_start_time + np.arange(len(event_sig)) * dt
    ax.plot(t_ev, event_sig, color='blue', linewidth=1.0, alpha=0.9,
            label=f'{sensor_a} event (template)')

    # Show where best match lands on B
    if lag_raw > 0:
        match_start = event_start_time + lag_raw
        ax.axvline(match_start, color='red', linewidth=1.5, linestyle='--',
                    label=f'Best match on {sensor_b} (raw)')
    if lag_env > 0 and abs(lag_env - lag_raw) > 0.01:
        match_start_e = event_start_time + lag_env
        ax.axvline(match_start_e, color='green', linewidth=1.5, linestyle='--',
                    label=f'Best match on {sensor_b} (envelope)')

    ax.set_title('Template (sensor A event) slid over sensor B signal', fontsize=11)
    ax.set_xlabel('Absolute time (seconds)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: Raw correlation curve
    ax = axes[1]
    if len(lags_raw) > 0:
        ax.plot(lags_raw, corr_raw, color='blue', linewidth=0.8)
        ax.axvline(lag_raw, color='red', linestyle='--', linewidth=1.5,
                    label=f'Peak: lag={lag_raw:.3f}s, corr={score_raw:.3f}')
        ax.legend(fontsize=9)
    ax.set_title('Sliding Correlation (raw signal)', fontsize=11)
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Lag (seconds)')
    ax.grid(True, alpha=0.3)

    # Bottom: Envelope correlation curve
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
    """Show the event from A overlaid on B at the matched position."""
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
        ax.text(0.5, 0.5, 'Matched segment extends beyond signal B',
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

def run(filepath, start_time, duration_mins, sensors, distance_m, threshold,
        window_sec=20.0, pre_trigger_sec=0.5, max_lag_sec=100.0, output_dir='graphs'):

    dt = 0.01  # 100 Hz

    print("=" * 60)
    print("  SIMPLE VEHICLE SPEED ESTIMATION")
    print("=" * 60)
    print(f"  Sensors:      {sensors}")
    print(f"  Distance:     {distance_m} m")
    print(f"  Threshold:    {threshold}")
    print(f"  Window:       {window_sec}s + {pre_trigger_sec}s pre-trigger")
    print(f"  Max lag:      {max_lag_sec}s")
    print()

    # --- Load ---
    df = load_data(filepath, sensors, start_time, duration_mins)

    # --- Stage 1: Extract events ---
    print("\n--- STAGE 1: Event Extraction ---")
    raw_signals = {}
    events_per_sensor = {}

    for sensor in sensors:
        raw = df[sensor].values.astype(np.float64)
        raw = raw - np.mean(raw)   # only preprocessing: DC removal
        raw_signals[sensor] = raw

        events = extract_events(raw, dt, threshold, window_sec, pre_trigger_sec)
        events_per_sensor[sensor] = events
        #print(events_per_sensor)

        print(f"  {sensor}: {len(events)} event(s)")
        for k, ev in enumerate(events):
            dur = ev['end_time'] - ev['start_time']
            print(f"    Event {k+1}: {ev['start_time']:.2f}s - {ev['end_time']:.2f}s "
                  f"({dur:.2f}s, {len(ev['trigger_times'])} triggers)")

    plot_stage1(raw_signals, events_per_sensor, sensors, dt, threshold,
                os.path.join(output_dir, 'stage1_events.png'))

    # --- Stage 2 and 3: For each consecutive sensor pair ---
    print("\n--- STAGE 2 and 3: Slide-Correlate + Speed ---")

    all_results = []

    for pair_idx in range(len(sensors) - 1):
        sensor_a = sensors[pair_idx]
        sensor_b = sensors[pair_idx + 1]
        events_a = events_per_sensor[sensor_a]
        full_b = raw_signals[sensor_b]

        print(f"\n  Pair: {sensor_a} -> {sensor_b} ({distance_m}m)")

        if not events_a:
            print(f"    No events on {sensor_a}")
            continue

        for i, ev_a in enumerate(events_a):
            print(f"\n    Event {i+1} from {sensor_a}:")
            print(f"      Time: {ev_a['start_time']:.2f}s - {ev_a['end_time']:.2f}s")

            # Slide this event over sensor B's full signal
            lag_raw, score_raw, lags_r, corr_r = slide_event_over_signal(
                ev_a['signal'], full_b, ev_a['start_idx'],
                dt, max_lag_sec, use_envelope=False
            )

            lag_env, score_env, lags_e, corr_e = slide_event_over_signal(
                ev_a['signal'], full_b, ev_a['start_idx'],
                dt, max_lag_sec, use_envelope=True
            )

            speed_raw = estimate_speed(lag_raw, distance_m)
            speed_env = estimate_speed(lag_env, distance_m)

            # Report
            print(f"      RAW:       corr={score_raw:.4f}  lag={lag_raw:.4f}s", end="")
            if 5 < speed_raw < 300:
                print(f"  speed={speed_raw:.1f} km/h")
            else:
                print(f"  speed={speed_raw:.1f} km/h (suspect)")

            print(f"      ENVELOPE:  corr={score_env:.4f}  lag={lag_env:.4f}s", end="")
            if 5 < speed_env < 300:
                print(f"  speed={speed_env:.1f} km/h")
            else:
                print(f"  speed={speed_env:.1f} km/h (suspect)")

            # Pick better method
            if score_env > score_raw:
                best_lag = lag_env
                best_score = score_env
                method = "envelope"
            else:
                best_lag = lag_raw
                best_score = score_raw
                method = "raw"

            best_speed = estimate_speed(best_lag, distance_m)
            print(f"      BEST:      {method} | lag={best_lag:.3f}s | "
                  f"speed={best_speed:.1f} km/h | corr={best_score:.3f}")

            all_results.append({
                'sensor_a': sensor_a, 'sensor_b': sensor_b,
                'event_idx': i, 'lag_raw': lag_raw, 'score_raw': score_raw,
                'lag_env': lag_env, 'score_env': score_env,
                'best_lag': best_lag, 'best_speed': best_speed,
                'best_method': method, 'best_score': best_score
            })

            # Plots
            tag = f"pair{pair_idx+1}_ev{i+1}"

            plot_stage2(
                ev_a['signal'], full_b, sensor_a, sensor_b,
                ev_a['start_time'], dt,
                lags_r, corr_r, lag_raw, score_raw,
                lags_e, corr_e, lag_env, score_env,
                os.path.join(output_dir, f'stage2_{tag}.png')
            )

            plot_stage3_aligned(
                ev_a['signal'], full_b, sensor_a, sensor_b,
                ev_a['start_idx'], best_lag, dt,
                os.path.join(output_dir, f'stage3_{tag}.png')
            )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if all_results:
        for r in all_results:
            print(f"  {r['sensor_a']}->{r['sensor_b']} event {r['event_idx']+1}: "
                  f"{r['best_speed']:.1f} km/h "
                  f"(lag={r['best_lag']:.3f}s, corr={r['best_score']:.3f}, "
                  f"method={r['best_method']})")
    else:
        print("  No events correlated.")
    print("=" * 60)

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Simple vehicle speed estimation')
    parser.add_argument('--path', required=True)
    parser.add_argument('--start_time', required=True)
    parser.add_argument('--duration_mins', type=float, required=True)
    parser.add_argument('--sensors', nargs='+', required=True,
                        help='Sensor columns in vehicle travel order')
    parser.add_argument('--distance', type=float, required=True,
                        help='Distance between consecutive sensors (meters)')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Amplitude threshold for event trigger')
    parser.add_argument('--window_sec', type=float, default=20.0)
    parser.add_argument('--pre_trigger_sec', type=float, default=0.5)
    parser.add_argument('--max_lag', type=float, default=5.0,
                        help='Max lag to search (seconds)')
    parser.add_argument('--output_dir', default='graphs')
    args = parser.parse_args()

    run(filepath=args.path, start_time=args.start_time,
        duration_mins=args.duration_mins, sensors=args.sensors,
        distance_m=args.distance, threshold=args.threshold,
        window_sec=args.window_sec, pre_trigger_sec=args.pre_trigger_sec,
        max_lag_sec=args.max_lag, output_dir=args.output_dir)


if __name__ == '__main__':
    main()


#ex: python3 simple_speed.py --path /Users/thomas/Data/Data_sensors/20250307/csv_acc/M001_2025-03-07_01-00-00_gg-112_int-2_th.csv
#  --start_time '2025/03/07 01:06:00' --duration_mins 4 --sensors 030911FF_x 030910F5_x --distance 727.25 --threshold 0.001 --max_lag 100.0