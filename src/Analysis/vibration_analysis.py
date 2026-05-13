import html
from turtle import width
from typing import Optional
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import psutil
import os
from scipy import signal
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter


def memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"Current memory usage: {mem:.2f} MB")


def find_csv_file(root_folder, date_str, hour):
    """Locate the CSV file for a given date and hour inside root_folder."""
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    csv_folder = os.path.join(root_folder, date_str, date_str, "csv_acc")
    for fname in os.listdir(csv_folder):
        if (fname.startswith(f"M001_{formatted_date}_{hour:02d}-00-00_")
                and f"_int-{hour+1}_" in fname and fname.endswith("_th.csv")):
            return os.path.join(csv_folder, fname)
    raise FileNotFoundError(f"No matching CSV file found for {date_str} hour {hour}")


def load_data_polars(filepath):
    """Load parquet file using Polars for memory efficiency"""
    print("Loading data with Polars...")
    memory_usage()
    # Load the data
    
    check1 = filepath.endswith('.csv')
    check2 = filepath.endswith('.parquet')
    if check1:
        df = pl.read_csv(filepath, separator= ';')
        print('reading the csv file:')
        # Count total rows without loading everything to memory
        total_rows = df.select(pl.len()).item() #use .collect if we use scan_csv function
        print(f"Total rows: {total_rows:,}")
    
        # Get column names for sensor data (assuming pattern ends with _z for vertical direction)
        sensor_columns = [col for col in df.columns if col != 'time']
        
        # Try to identify time column
        time_column_candidates = ['time', 'timestamp', 'date']
        time_column = next((col for col in df.columns if col in time_column_candidates), None)
        print(f"Found {len(sensor_columns)} sensor columns:")
        print(f"Using '{time_column}' as the time column")
    if check2:
        df = pl.scan_parquet(filepath)    
        # Count total rows without loading everything to memory
        total_rows = df.select(pl.len()).collect().item()
        print(f"Total rows: {total_rows:,}")
        
        # Get column names for sensor data (assuming pattern ends with _z for vertical direction)
        sensor_columns = [col for col in df.collect_schema().keys() if col != 'time']
        
        # Try to identify time column
        time_column_candidates = ['time', 'timestamp', 'date']
        time_column = next((col for col in df.collect_schema().keys() if col.lower() in time_column_candidates), None)
        
        print(f"Found {len(sensor_columns)} sensor columns:")
        print(f"Using '{time_column}' as the time column")
    
    memory_usage()
    return df, sensor_columns, time_column

def parse_sensor_ids(sensor_str, available_sensors):
    if sensor_str is None:
           return available_sensors
    
    # Parse comma-separated values
    requested = [s.strip() for s in sensor_str.split(',')]

    # Validate each sensor exists
    invalid = [s for s in requested if s not in available_sensors]
    if invalid:
        raise ValueError(
            f"Invalid sensor IDs: {invalid}\n"
            f"Available sensors: {available_sensors}"
        )
    return requested

# =========================================================
# THRESHOLD EXTRACTION
# =========================================================
def extract_sensor_thresholds(bridge, sensor_ids):
    """Extract sensor-specific trigger thresholds from bridge model."""
    sensor_thresholds = {}
    
    for sensor_id in sensor_ids:
        sensor = bridge[sensor_id]  # Sensor object
        sensor_thresholds[sensor_id] = sensor.trigger_threshold
    
    return sensor_thresholds

def _get_filtered_mask(sensor_series: pd.Series, threshold: float, sample_period: int) -> np.ndarray:
    """
    Return a boolean mask of the same length as sensor_series.
    True  → sample is inside an active window (|value| >= threshold,
            plus the sample_period tail after each crossing).
    False → sample is outside every active window.
    """
    n    = len(sensor_series)
    mask = np.zeros(n, dtype=bool)
    vals = sensor_series.to_numpy()   # work on a plain numpy array for speed

    i = 0
    while i < n:
        if np.abs(vals[i]) >= threshold:
            start = i
            end   = min(i + sample_period, n)

            # Extend the window as long as the signal keeps crossing threshold
            while i < end:
                if np.abs(vals[i]) >= threshold:
                    end = min(i + sample_period, n)
                i += 1

            mask[start:end] = True
        else:
            i += 1
    
    return mask


def filterby_threshold(
    data:          pd.DataFrame,
    threshold:     float,
    sample_period: int,
    sensor_columns: "str | list[str]"):
    """
    Apply threshold-based event detection to one or more sensor columns.
    """
    # ── Normalise to list ────────────────────────────────────────────────────
    if isinstance(sensor_columns, str):
        sensor_columns = [sensor_columns]

    # ── Output DataFrame – start with the time column ────────────────────────
    filtered_df = pd.DataFrame({'time': data['time'].to_numpy()})
    ratios: dict[str, float] = {}

    # ── Process each sensor independently ───────────────────────────────────
    for col in sensor_columns:
        sensor_series = pd.Series(data[col].to_numpy() if hasattr(data[col], 'to_numpy') else data[col])  # guarantee 0-based index
        mask          = _get_filtered_mask(sensor_series, threshold, sample_period)

        # Build filtered signal: keep value where mask is True, else 0
        filtered_vals = sensor_series.to_numpy().copy().astype(float)
        filtered_vals[~mask] = 0.0

        filtered_df[col]                = filtered_vals
        filtered_df[f'{col}_original']  = sensor_series.to_numpy()
        ratios[col]                     = mask.sum() / len(mask)

    # ── Plot: one subplot per sensor, 2-column grid ──────────────────────────
    n    = len(sensor_columns)
    cols = min(2, n)
    rows = -(-n // cols)   # ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 4 * rows), sharex=False)
    axes = np.array(axes).flatten()

    for i, col in enumerate(sensor_columns):
        ax       = axes[i]
        time_arr = filtered_df['time']
        original = filtered_df[f'{col}_original']
        filtered = filtered_df[col]

        ax.plot(time_arr, original, label='original', linewidth=0.8, alpha=0.6)
        ax.plot(time_arr, filtered, label='filtered',
                color='y', linewidth=1.0, alpha=0.85)
        ax.axhline( threshold, color='red', linestyle='--', linewidth=0.9, label='threshold')
        ax.axhline(-threshold, color='red', linestyle='--', linewidth=0.9)

        ax.set_title(f'{col}   (retained: {ratios[col]:.1%})', fontsize=11)
        ax.set_xlabel('Time',         fontsize=10)
        ax.set_ylabel('Acceleration', fontsize=10)
        ax.set_ylim(-0.006, 0.006)
        ax.tick_params(axis='x', labelsize=8, rotation=30)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        

    # Hide unused panels when n is not a perfect multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Threshold Filtering – All Sensors', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return filtered_df, ratios


def filter_dc_by_mean(df: pl.DataFrame, sensor_columns: list[str]) -> pl.DataFrame:
  
    # Clone the DataFrame to avoid modifying the original
    result_df = df.clone()
    
    # Process only specified sensor columns, ignoring 'time'
    for col in sensor_columns:
            # Compute the mean as a scalar
            col_mean = df.select(pl.col(col).mean()).item() #use .collect.item() if we scan_csv or prquet
            
            # Subtract mean and update the column
            result_df = result_df.with_columns(
                (pl.col(col) - col_mean).alias(col)
            )


    return result_df

def get_only_interested_duration(df, sensor_columns, time_column, start_time, duration_mins):
    memory_usage()

    # Select only necessary columns and sample at specified interval
    sampled_df = (df.select([time_column] + sensor_columns)  # Keep relevant columns
    #.collect() used if you are scanning the file
    .to_pandas()
    )
    print(f"Sampled data shape: {sampled_df.shape}")
    memory_usage()

    sampled_df[time_column] = pd.to_datetime(sampled_df[time_column], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    

    #PLOT ONLY duration we want to analyse 
    start_time = pd.to_datetime(start_time)
    

    end_time = start_time + pd.Timedelta(minutes=duration_mins)

    #limit to the specific time frame
    sampled_df = sampled_df[(sampled_df[time_column]>=start_time)&(sampled_df[time_column]<=end_time)]
    
    memory_usage()

    return sampled_df

# ============================================================================
# PEAK DETECTION
# ============================================================================
 
def find_sensor_peaks(
    df: pd.DataFrame,
    sensor_ids: str,
    time_column: str,
    sensor_thresholds: float,
    sample_period
):

    results = {}
 
    for sensor_id in sensor_ids:
        sensor_series = pd.Series(df[sensor_id].values, dtype=float)
        raw_signal    = sensor_series.to_numpy()
        time_series   = df[time_column]               # pandas Series
        events        = []

         # Get sensor-specific threshold
        threshold = sensor_thresholds.get(sensor_id, 0.002)

        #print(threshold)
 
        # Step 1: get event mask using existing _get_filtered_mask
        mask = _get_filtered_mask(sensor_series, threshold, sample_period)
        
        
        # Step 2: find contiguous True regions (each = one event window)
        diff   = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1       # False->True transitions
        ends   = np.where(diff == -1)[0] + 1       # True->False transitions
 
        # Handle edge cases: mask starts or ends with True
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(mask))
 
        # Step 3: within each event window, find extrema on RAW signal
        for win_start, win_end in zip(starts, ends):
            seg = raw_signal[win_start:win_end]
            seg_len = len(seg)
 
            # --- Collect all extrema (local max + local min) on raw signal ---
            extrema = []
 
            if (seg[0] > 0 and seg[0] > seg[1]) or \
               (seg[0] < 0 and seg[0] < seg[1]):
                extrema.append(0)
 
            # k=1..seg_len-2: interior extrema
            for k in range(1, seg_len - 1):
                if seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1]:
                    extrema.append(k)   # local max
                elif seg[k] <= seg[k - 1] and seg[k] <= seg[k + 1]:
                    extrema.append(k)   # local min
 
            # Fallback: no extrema found (monotonic signal), use sample
            # with largest absolute amplitude
            if not extrema:
                fallback = int(np.argmax(np.abs(seg)))
                extrema.append(fallback)
 
            # --- First peak: first extremum in the window ---
            first_peak_local = extrema[0]
 
            # --- Dominant peak: extremum with largest |amplitude| ---
            dominant_local = max(extrema, key=lambda idx: abs(seg[idx]))
 
            # Convert to global indices, return signed amplitudes
            fp_idx = win_start + first_peak_local
            dp_idx = win_start + dominant_local
 
            events.append((
                time_series.iloc[fp_idx],          # first_peak_time
                time_series.iloc[dp_idx],          # dominant_peak_time
                float(raw_signal[fp_idx]),         # first_peak_amplitude (signed)
                float(raw_signal[dp_idx]),         # dominant_peak_amplitude (signed)
            ))
 
        results[sensor_id] = events
        print(f"[PEAKS] {sensor_id}: {len(events)} event(s) "
              f"(threshold={threshold}, sample_period={sample_period})")
 
    return results


def visualize_all_sensors(sampled_df, sensor_columns, time_column, start_time, duration_mins):
    print(f"Visualizing sensors in one campate with sample interval from {start_time} to {duration_mins} mins...")
    memory_usage()
 
    # Choose the sensors you want to plot with matplolib
    sensor_list = sensor_columns[:8]
    n = len(sensor_list)
    cols = 4
    rows = -(-n // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True)
    axes = axes.flatten()

    for i, sensor in enumerate(sensor_list):
        ax = axes[i]
        ax.plot(sampled_df[time_column], sampled_df[sensor], linewidth=1)
        ax.set_title(sensor, fontsize=10)
        ax.tick_params(axis='x', labelrotation=45)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Sensor Vibration Plots (Vertical Direction)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
   
    fig = go.Figure()
    sensor_list = sensor_columns[:6]

    n = len(sensor_list)
    cols = 3
    rows = -(-n // cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=sensor_list
    )
    for i, sensor in enumerate(sensor_list):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        fig.add_trace(
            go.Scatter(
                x=sampled_df[time_column],
                y=sampled_df[sensor],
                mode='lines',
                name=sensor,
                line=dict(width=1),
                opacity=0.8
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title_text="Sensor Vibration Plots (Vertical Direction)",
        showlegend=False
    )
    fig.show()
    print("All sensors visualization saved to all_sensors_acceleration.png")
    memory_usage()
    return sampled_df

def multi_sensor_spectrogram(df, sensor_columns, cols=3):
    """
    Plot spectrograms for multiple sensors in subplots.
    """
    sensor_list = sensor_columns[:6]
    n = len(sensor_list)
    rows = -(-n // cols)  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = axes.flatten()

    for i, sensor_column in enumerate(sensor_list):
        x = df[sensor_column].values
        f, t, Sxx = signal.spectrogram(
            x,
            fs= 100,
            window=signal.get_window('hamming', 256),
            nperseg=256,
            noverlap=64
        )
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        ax = axes[i]
        pcm = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='viridis')
        ax.set_title(sensor_column, fontsize=10)
        ax.set_ylabel('Freq [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylim([0, 50])
        fig.colorbar(pcm, ax=ax, label='Power [dB]')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Spectrograms of Selected Sensors', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]
    plt.suptitle(f"Spectrogram of {sensor_column}\nStart: {start_time} | End: {end_time}")
    plt.show()

def visualize_sensor_histograms(df, sensor_columns, bins=50):
    """Create histograms for each sensor to analyze distribution"""
    print("Creating histograms for each sensor...")
    memory_usage()
    
    grid_size = int(np.ceil(np.sqrt(len(sensor_columns))))
    
    plt.figure(figsize=(16, 16))

    for i, sensor in enumerate(sensor_columns, 1):
        plt.subplot(grid_size, grid_size, i)
        plt.hist(df[sensor], bins=bins, alpha=0.7)
        plt.title(sensor)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print("Sensor histograms saved to sensor_histograms.png")
    memory_usage()

def waterfall_3d_plot(df, sensor_columns, time_column, fs=100, downsample_step=2,
                      save_path='waterfall_3d.png'):
    """
    3D Waterfall plot showing vehicle-induced acceleration across multiple bridge sensors.
    """
    print("Generating 3D Waterfall plot...")
    memory_usage()

    sensor_list = sensor_columns[:8]
    n_sensors   = len(sensor_list)

    if n_sensors == 0:
        print("No sensor columns available for waterfall plot.")
        return

    t_raw = df[time_column].values
    if hasattr(t_raw[0], 'timestamp'):
        t_sec = np.array([(ts - t_raw[0]).total_seconds() for ts in t_raw])
    else:
        t_sec = (t_raw - t_raw[0]).astype('timedelta64[ms]').astype(float) / 1000.0

    step   = max(1, downsample_step)
    t_ds   = t_sec[::step]

    signals = []
    for col in sensor_list:
        sig = df[col].values[::step].astype(float)
        signals.append(sig)

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / max(n_sensors - 1, 1)) for i in range(n_sensors)]

    y_positions = np.arange(n_sensors, dtype=float)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for i in range(n_sensors - 1, -1, -1):
        y   = y_positions[i]
        sig = signals[i]
        col = colors[i]

        verts_x = np.concatenate([[t_ds[0]], t_ds, [t_ds[-1]]])
        verts_z = np.concatenate([[0],        sig,  [0]])
        verts_y = np.full_like(verts_x, y)

        verts = [list(zip(verts_x, verts_y, verts_z))]
        poly  = Poly3DCollection(verts, alpha=0.35, zorder=i)
        poly.set_facecolor((*col[:3], 0.20))
        poly.set_edgecolor((*col[:3], 0.0))
        ax.add_collection3d(poly)

        ax.plot(t_ds, [y] * len(t_ds), sig,
                color=col, linewidth=0.9, alpha=0.95, zorder=i + n_sensors)

    ax.set_xlabel("Elapsed Time  (s)",        color="black", fontsize=10, labelpad=10)
    ax.set_ylabel("Sensor",                   color="black", fontsize=10, labelpad=14)
    ax.set_zlabel("Acceleration  (m/s²)",     color="black", fontsize=10, labelpad=8)
    ax.set_title("Bridge Accelerometers – 3D Waterfall (Vehicle Pass)",
                 color="black", fontsize=13, fontweight="bold", pad=18)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sensor_list, fontsize=7, color="black")

    ax.set_xlim(t_ds[0], t_ds[-1])
    ax.set_ylim(y_positions[0] - 0.5, y_positions[-1] + 0.5)

    ax.tick_params(colors="black", labelsize=7.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#cccccc")

    ax.xaxis.line.set_color("#cccccc")
    ax.yaxis.line.set_color("#cccccc")
    ax.zaxis.line.set_color("#cccccc")
    ax.grid(True, color="#dddddd", linewidth=0.4)

    sm = plt.cm.ScalarMappable(
        cmap="tab10",
        norm=plt.Normalize(0, n_sensors - 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=18)
    cbar.set_label("Sensor index (front → back)", color="black", fontsize=8)
    cbar.set_ticks(np.arange(n_sensors))
    cbar.set_ticklabels(
        [s[:8] for s in sensor_list],
        fontsize=6
    )
    cbar.ax.yaxis.set_tick_params(colors="black", labelsize=6)
    cbar.outline.set_edgecolor("#cccccc")

    ax.view_init(elev=28, azim=-55)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    print(f"3D Waterfall plot saved to: {save_path}")
    plt.show()
    memory_usage()

def visualize_overlay(df: pd.DataFrame, sensor_ids, time_column, peak_data, output_path=None) -> None:

    invalid = [s for s in sensor_ids if s not in df.columns]
    if invalid:
        raise ValueError(f"Sensors not found in data: {invalid}")
 
    fig = go.Figure()
 
    for sensor_id in sensor_ids:
        fig.add_trace(go.Scatter(
            x=df[time_column],
            y=df[sensor_id],
            mode='lines',
            name=sensor_id,
            line=dict(width=1.2),
            opacity=0.85,
            legendgroup=sensor_id
        ))

        if peak_data and sensor_id in peak_data:
            events = peak_data[sensor_id]
            if not events:
                continue
 
            fp_times = [e[0] for e in events]
            dp_times = [e[1] for e in events]
            fp_amps  = [e[2] for e in events]
            dp_amps  = [e[3] for e in events]
 
            fig.add_trace(go.Scatter(
                x=fp_times,
                y=fp_amps,
                mode='markers',
                name=f'{sensor_id} first peak',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    line=dict(width=1, color='white'),
                ),
                legendgroup=sensor_id,
                showlegend=False,
                hovertemplate=(
                    'First Peak<br>'
                    'Time: %{x}<br>'
                    'Amplitude: %{y:.6f}<br>'
                    '<extra></extra>'
                ),
            ))
 
            fig.add_trace(go.Scatter(
                x=dp_times,
                y=dp_amps,
                mode='markers',
                name=f'{sensor_id} dominant peak',
                marker=dict(
                    symbol='star',
                    size=10,
                    line=dict(width=1, color='white'),
                ),
                legendgroup=sensor_id,
                showlegend=False,
                hovertemplate=(
                    'Dominant Peak<br>'
                    'Time: %{x}<br>'
                    'Amplitude: %{y:.6f}<br>'
                    '<extra></extra>'
                ),
            ))
 
    plot_title = f"Acceleration Data from Sensors ({len(sensor_ids)} sensors)"
 
    fig.update_layout(
        title=plot_title,
        xaxis_title="Time",
        yaxis_title="Acceleration",
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(
            title="Sensors",
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
        ),
        height=900,
    )
 
    if output_path:
        fig.write_html(f'/home/thomas/graphs/fig{sensor_id}.html')
    fig.write_html(f'/home/thomas/graphs/fig_{sensor_id}.html')
    fig.show()


def main():
    parser = argparse.ArgumentParser('Analyse the vibration relating the dynamic weighing data')
    parser.add_argument('--root_folder',   type=str, required=True, help='Root folder containing sensor CSV data')
    parser.add_argument('--start_time',    type=str, required=True, help='Starting time frame of interest (YYYY/MM/DD HH:MM:SS)')
    parser.add_argument('--duration_mins', type=float, required=True, help='Duration in minutes of time frame of interest')
    parser.add_argument('--sensor',        type=str, default=None,  help='Sensor ID(s) to analyze (comma-separated for multiple)')

    args = parser.parse_args()
    root_folder   = args.root_folder
    start_time    = args.start_time
    duration_mins = args.duration_mins
    sensor_arg    = args.sensor

    sample_period = 500

    # ── Derive CSV path from start_time and root_folder ──────────────────────
    start_dt = pd.to_datetime(start_time)
    date_str = start_dt.strftime("%Y%m%d")
    hour     = start_dt.hour

    print(f"Searching for CSV file: date={date_str}, hour={hour:02d}...")
    path = find_csv_file(root_folder, date_str, hour)
    print(f"Found: {path}")

    # ── Load bridge model ─────────────────────────────────────────────────────
    print("Loading bridge model...")
    bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    junctions = bridge.find_boundaries()
    
    boundary_sensors = [s for j in junctions for s in j.sensor_ids()]
    boundary_sensors = list(dict.fromkeys(boundary_sensors))  # deduplicate, preserve order
    
    print(f"Boundary sensors (after deduplication): {len(boundary_sensors)}")
    
    print("\nExtracting sensor-specific trigger thresholds...")
    sensor_thresholds = extract_sensor_thresholds(bridge, boundary_sensors)
    
    threshold_values = list(sensor_thresholds.values())
    print(f"Loaded {len(sensor_thresholds)} sensor thresholds")
    print(f"Threshold range: {min(threshold_values):.6f} to {max(threshold_values):.6f}")
    print(f"Threshold mean: {np.mean(threshold_values):.6f}")

    # ── Load data ─────────────────────────────────────────────────────────────
    df, available_sensors, time_column = load_data_polars(path)

    if sensor_arg is None:
        sensor_columns = boundary_sensors
    else:
        sensor_columns = parse_sensor_ids(sensor_arg, available_sensors)

    # ── Process ───────────────────────────────────────────────────────────────
    no_dc_df   = filter_dc_by_mean(df, sensor_columns)
    sampled_df = get_only_interested_duration(no_dc_df, sensor_columns, time_column, start_time, duration_mins)
    peak_data  = find_sensor_peaks(sampled_df, sensor_columns, time_column, sensor_thresholds, sample_period)

    visualize_overlay(sampled_df, sensor_columns, time_column, peak_data)

    print("Analysis complete!")
    memory_usage()

if __name__ == "__main__":
    main()
    # Instructions to run this script:
    # python3 vibration_analysis.py \
    #   --root_folder /Users/thomas/Data/Data_sensors \
    #   --start_time '2025/03/07 01:05:00' \
    #   --duration_mins 5 \
    #   --sensor 03091203_x
    #
    # For all boundary sensors (omit --sensor):
    # python3 vibration_analysis.py \
    #   --root_folder /Users/thomas/Data/Data_sensors \
    #   --start_time '2025/03/03 00:00:00' \
    #   --duration_mins 5