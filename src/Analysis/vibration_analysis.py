import html
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

def memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"Current memory usage: {mem:.2f} MB")

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
    
    # # sensors 99,100,101
    # campate2_sensor_columns = ['030911D2_x', '03091005_x', '0309101F_x']
    # #sensors in order 53->52->51
    # campate1b_sensor_columns = ['0309100F_x', '030910F6_x', '0309101E_x']
    # #sensors in order 106->105->104
    # campate1a_sensor_columns = ['030911FF_x', '030911EF_x', '03091200_x', '03091155_z', '03091207_x', '03091119_z'] 
    # #sensor column on whole brideg
    # all_campate_sensor_columns = ['030911FF_x', '03091017_z', '03091113_x', '0309123B_z', '03091111_z', '03091003_x'] 

    # first_campata = [ '030911FF_x', '030911EF_x', '03091200_x', '03091155_z', '0309100F_x', '030910F6_x', '0309101E_x', '03091018_z']

    # # whole bridge overview: 106->93->83->78->67->54
    # whole_bridge_overview = [ '030911FF_x', '0309113F_z', '0309123B_z', '03091204_x', '03091111_z', '03091003_x']
    #  #first campate sensors in order 106->105->104->53->52->51
    # #sensor_columns = [col for col in campate1a_sensor_columns if col in df.columns]  
    # sensor_columns = [col for col in whole_bridge_overview if col in df.columns]
    
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
    #plt.savefig('ETFA_multisensor_whole_bridge_overview.png', dpi=150)
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
    
    #print('intereseted time dataframe-start', sampled_df.head(1))
    #print('limited time frame dataframe-end', sampled_df.tail(1))
    memory_usage()

    return sampled_df

# ============================================================================
# PEAK DETECTION
# ============================================================================
 
def find_sensor_peaks(
    df: pd.DataFrame,
    sensor_ids: str,
    time_column: str,
    threshold: float,
    sample_period
):
    """
    Find one first peak and one dominant peak per event window.
 
    Uses _get_filtered_mask() to group nearby threshold crossings into
    single event windows (via sample_period extension), then within each
    contiguous window finds extrema on the RAW signal (not abs):
      - first_peak:    first extremum (local max or min) in the window,
                       including the very first sample (boundary check)
      - dominant_peak: extremum with the largest absolute amplitude
 
    Peak detection uses raw signed values so that a positive peak followed
    by a larger negative swing doesn't get missed.
 
    Args:
        df: DataFrame with time and sensor columns (DC already removed)
        sensor_ids: List of sensor column names
        time_column: Name of time column
        threshold: Absolute amplitude threshold for event window detection
        sample_period: Samples to extend window after each crossing
                       (same parameter as filterby_threshold, default 300)
 
    Returns:
        Dict mapping sensor_id -> list of
        (first_peak_time, dominant_peak_time, first_peak_amplitude, dominant_peak_amplitude)
    """
    results = {}
 
    for sensor_id in sensor_ids:
        sensor_series = pd.Series(df[sensor_id].values, dtype=float)
        raw_signal    = sensor_series.to_numpy()
        time_series   = df[time_column]               # pandas Series
        events        = []
 
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
 
            # if seg_len < 2:
            #     # Single sample — use it directly
            #     p_idx = win_start
            #     events.append((
            #         time_series.iloc[p_idx],
            #         time_series.iloc[p_idx],
            #         float(raw_signal[p_idx]),
            #         float(raw_signal[p_idx]),
            #     ))
            #     continue
 
            # --- Collect all extrema (local max + local min) on raw signal ---
            extrema = []
 
            # k=0: boundary peak — signal already turning at window start.
            # Positive crossing that immediately drops: seg[0] > 0 and seg[0] > seg[1]
            # Negative crossing that immediately rises: seg[0] < 0 and seg[0] < seg[1]
            # If signal is still growing in magnitude, k=0 is NOT a peak.
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
    sensor_list = sensor_columns[:8]  # or list(vertical_columns.values())[:6]
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    #plt.savefig('graphs/subplots.png')
    plt.show()

    # Save to file
    #fig.savefig(f'src/results/Interactive_vibrations/vibration_data_{start_time} - {duration_mins} mins.png', dpi=300)
        
    fig = go.Figure()
   # Choose the sensors you want to plot
    sensor_list = sensor_columns[:6] # Adjust the number as needed

    # Create subplot structure: 2 rows, 3 columns for 6 sensors
    n = len(sensor_list)
    cols = 3
    rows = -(-n // cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=sensor_list  # Use sensor names as titles
    )
    # Loop and add each sensor's trace to the appropriate subplot
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

    # Update layout
    fig.update_layout(
        title_text="Sensor Vibration Plots (Vertical Direction)",
        showlegend=False
    )
    # Display the plot
    fig.show()
    #fig.write_html(f'src/results/Interactive_vibrations/vibration_data_{start_time} - {duration_mins} mins.html')
    print("All sensors visualization saved to all_sensors_acceleration.png")
    memory_usage()
    return sampled_df

def multi_sensor_spectrogram(df, sensor_columns, cols=3):
    """
    Plot spectrograms for multiple sensors in subplots.

    Parameters:
    - df: DataFrame containing the sensor data.
    - sensor_list: List of column names (sensors) to plot.
    - cols: Number of columns for subplot layout.
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
            fs= 100, #sampling_rate,
            window=signal.get_window('hamming', 256),
            nperseg=256,
            noverlap=64
        )
        # Convert power to decibels (dB)
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)   # add epsilon to avoid log(0)
        ax = axes[i]
        pcm = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='viridis')
        ax.set_title(sensor_column, fontsize=10)
        ax.set_ylabel('Freq [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylim([0, 50])
        fig.colorbar(pcm, ax=ax, label='Power [dB]')

    # Remove unused axes if sensor count is not a perfect multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Spectrograms of Selected Sensors', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]
    plt.suptitle(f"Spectrogram of {sensor_column}\nStart: {start_time} | End: {end_time}")
    #plt.savefig('graphs/multisensor_spectogram.png')
    plt.show()

def visualize_sensor_histograms(df, sensor_columns, bins=50):
    """Create histograms for each sensor to analyze distribution"""
    print("Creating histograms for each sensor...")
    memory_usage()
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(sensor_columns))))
    
    # Create figure
    plt.figure(figsize=(16, 16))

    for i, sensor in enumerate(sensor_columns, 1):
        plt.subplot(grid_size, grid_size, i)
        plt.hist(df[sensor], bins=bins, alpha=0.7)
        plt.title(sensor)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('sensor_histograms.png', dpi=300)
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

    sensor_list = sensor_columns[:8]          # cap at 6 sensors for readability
    n_sensors   = len(sensor_list)

    if n_sensors == 0:
        print("No sensor columns available for waterfall plot.")
        return

    # ── Build time axis in seconds ────────────────────────────────────────────
    # Use elapsed seconds from the first timestamp so the X-axis is readable
    t_raw = df[time_column].values
    if hasattr(t_raw[0], 'timestamp'):          # datetime objects
        t_sec = np.array([(ts - t_raw[0]).total_seconds() for ts in t_raw])
    else:                                        # numpy datetime64
        t_sec = (t_raw - t_raw[0]).astype('timedelta64[ms]').astype(float) / 1000.0

    # Downsample for speed
    step   = max(1, downsample_step)
    t_ds   = t_sec[::step]

    # ── Collect signals ───────────────────────────────────────────────────────
    signals = []
    for col in sensor_list:
        sig = df[col].values[::step].astype(float)
        signals.append(sig)

    # ── Colour map: one colour per sensor (tab10, front → back) ─────────────
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / max(n_sensors - 1, 1)) for i in range(n_sensors)]

    # ── Y-axis: integer positions for sensors, labelled with sensor names ─────
    # We space sensors 1 unit apart on the Y-axis; labels replace the numbers
    y_positions = np.arange(n_sensors, dtype=float)

    # ── Figure (white theme) ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for i in range(n_sensors - 1, -1, -1):     # draw back-to-front for overlap
        y   = y_positions[i]
        sig = sig
        col = colors[i]

        # Closed polygon path for the waterfall ribbon
        verts_x = np.concatenate([[t_ds[0]], t_ds, [t_ds[-1]]])
        verts_z = np.concatenate([[0],        sig,  [0]])
        verts_y = np.full_like(verts_x, y)

        verts = [list(zip(verts_x, verts_y, verts_z))]
        poly  = Poly3DCollection(verts, alpha=0.35, zorder=i)
        poly.set_facecolor((*col[:3], 0.20))
        poly.set_edgecolor((*col[:3], 0.0))
        ax.add_collection3d(poly)

        # Solid line on top of the ribbon
        ax.plot(t_ds, [y] * len(t_ds), sig,
                color=col, linewidth=0.9, alpha=0.95, zorder=i + n_sensors)

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax.set_xlabel("Elapsed Time  (s)",        color="black", fontsize=10, labelpad=10)
    ax.set_ylabel("Sensor",                   color="black", fontsize=10, labelpad=14)
    ax.set_zlabel("Acceleration  (m/s²)",     color="black", fontsize=10, labelpad=8)
    ax.set_title("Bridge Accelerometers – 3D Waterfall (Vehicle Pass)",
                 color="black", fontsize=13, fontweight="bold", pad=18)

    # Replace numeric Y ticks with sensor names
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

    # Colour bar mapping sensor index → sensor name
    sm = plt.cm.ScalarMappable(
        cmap="tab10",
        norm=plt.Normalize(0, n_sensors - 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=18)
    cbar.set_label("Sensor index (front → back)", color="black", fontsize=8)
    cbar.set_ticks(np.arange(n_sensors))
    cbar.set_ticklabels(
        [s[:8] for s in sensor_list],   # truncate long names for the colour bar
        fontsize=6
    )
    cbar.ax.yaxis.set_tick_params(colors="black", labelsize=6)
    cbar.outline.set_edgecolor("#cccccc")

    ax.view_init(elev=28, azim=-55)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    #plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"3D Waterfall plot saved to: {save_path}")
    plt.show()
    memory_usage()

def visualize_overlay(df: pd.DataFrame,sensor_ids, time_column, peak_data, output_path =None) -> None:

    invalid = [s for s in sensor_ids if s not in df.columns]
    if invalid:
        raise ValueError(f"Sensors not found in data: {invalid}")
 
 
    fig = go.Figure()

     # Assign colors so peak markers match their sensor trace
    # colors = [
    #     '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    #     '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
    # ]

    #print(peak_data)
 
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

         
        # Peak markers (if available)
        if peak_data and sensor_id in peak_data:
            events = peak_data[sensor_id]
            if not events:
                continue
 
            fp_times = [e[0] for e in events]   # first_peak_time
            dp_times = [e[1] for e in events]   # dominant_peak_time
            fp_amps  = [e[2] for e in events]   # first_peak_amplitude
            dp_amps  = [e[3] for e in events]   # dominant_peak_amplitude
 
            #print(fp_times)
            # First peaks — triangle markers
            fig.add_trace(go.Scatter(
                x=fp_times,
                y=fp_amps,
                mode='markers',
                name=f'{sensor_id} first peak',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    #color=color,
                    line=dict(width=1, color='white'),
                ),
                legendgroup=sensor_id,
                showlegend=True,
                hovertemplate=(
                    'First Peak<br>'
                    'Time: %{x}<br>'
                    'Amplitude: %{y:.6f}<br>'
                    '<extra></extra>'
                ),
            ))
 
            # Dominant peaks — star markers
            fig.add_trace(go.Scatter(
                x=dp_times,
                y=dp_amps,
                mode='markers',
                name=f'{sensor_id} dominant peak',
                marker=dict(
                    symbol='star',
                    size=10,
                   # color=color,
                    line=dict(width=1, color='white'),
                ),
                legendgroup=sensor_id,
                showlegend=True,
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
        height=600,
    )
 
    if output_path:
       fig.write_html(f'/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Graphs/first_peak/fig{sensor_id}.html')
       pass
    fig.write_html(f'/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Graphs/first_peak/fig{sensor_id}.html')
    fig.show()


def main():
    parser = argparse.ArgumentParser('Analyse the vibration realting the dyanmic weighing data')
    parser.add_argument('--path', type = str, required=True, help= 'path for the file')
    parser.add_argument('--start_time', type=str, required=True, help= 'starting time frame interedted in')
    parser.add_argument('--duration_mins', type=float, required=True, help = 'duration in mins of time frame interested')
    parser.add_argument('--sensor', type=str, default=None, help = 'Sensor ID(s) to analyze (comma-separated for multiple)')

    #ex for script RUNNING: python3 vibration_analysis.py --path  --start_time --duration_mins
    args = parser.parse_args()
    path = args.path # Path to your parquet file
    start_time = args.start_time 
    duration_mins = args.duration_mins
    sensor_columns = args.sensor
    threshold = 0.002
    sample_period = 300
      
    # Load data using Polars
    df, available_sensors, time_column = load_data_polars(path)

    sensor_columns = parse_sensor_ids(sensor_columns, available_sensors)
    
    # Process the filtered data
    no_dc_df = filter_dc_by_mean(df, sensor_columns)

    sampled_df = get_only_interested_duration(no_dc_df, sensor_columns, time_column, start_time, duration_mins)

    peak_data = find_sensor_peaks(sampled_df, sensor_columns, time_column, threshold, sample_period)

    visualize_overlay(sampled_df,sensor_columns, time_column, peak_data)


    # visualise each sensor in campate for a sample interval
    #sampled_df = visualize_all_sensors(, sensor_columns, time_column, start_time, duration_mins)

    #filtered_df, signal_fil_ratio = filterby_threshold(sampled_df, threshold, sample_period, sensor_columns)
    
    #multi_sensor_spectrogram(sampled_df, sensor_columns, cols=3)

    #visualize_sensor_histograms(sampled_df, sensor_columns, bins=50)

    # waterfall_3d_plot(
    #         df=filtered_df,
    #         sensor_columns=sensor_columns,
    #         time_column=time_column,
    #         fs=100,
    #         downsample_step=2,
    #         save_path='waterfall_3d_whole_bridge_overview.png'
    #     )

   
    print("Analysis complete!")
    memory_usage()

if __name__ == "__main__":
    main()
    #instructions to run this parametric scripts:
    #check wether the parameters correctly matching the format(for ex: the date and month should be interchanegd from the format of weighing data)  
    #  python3 vibration_analysis.py --path /Users/thomas/Data/Data_sensors/20250307/csv_acc/M001_2025-03-07_01-00-00_gg-112_int-2_th.csv --start_time '2025/03/07 01:05:00' --duration_mins 5 --sensor 03091203_x,0309101F_x