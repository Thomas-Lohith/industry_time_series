"""
Vehicle Event Analysis: Extract peak timestamps from boundary sensor pairs

For each vehicle event:
1. Adjust timestamp (add offset for weighing system delay)
2. Find and load appropriate hourly CSV file
3. Extract time window around vehicle passage
4. Detect first peak and dominant peak for each boundary sensor
5. Output sensor pair timestamps to CSV

Output CSV structure:
- Rows: vehicle events (v1, v2, ...)
- Columns: for each sensor pair, 4 timestamps:
    - sensor1_first_peak_time
    - sensor1_peak_time
    - sensor2_first_peak_time
    - sensor2_peak_time
"""

from pickle import TRUE
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

# Import from provided code

from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter


# =========================================================
# FILE DISCOVERY
# =========================================================
def find_csv_file(root_folder, date_str, hour):
    """Find the hourly CSV file for given date and hour."""
    hour1 = hour + 1
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    print(formatted_date)
    csv_folder = os.path.join(root_folder, date_str, "csv_acc")

    for fname in os.listdir(csv_folder):
        if (
            fname.startswith(f"M001_{formatted_date}_{hour:02d}-00-00_")
            and f"_int-{hour1}_" in fname
            and fname.endswith("_th.csv")
        ):
            return os.path.join(csv_folder, fname)

    raise FileNotFoundError(f"No matching CSV file found for {date_str} hour {hour}")


# =========================================================
# DATA EXTRACTION
# =========================================================
def get_only_interested_duration(df, sensor_columns, time_column, start_time, duration_mins):
    """Extract specific time window from dataframe."""
    print(f"  Selecting columns and converting to pandas...")
    
    # Select only necessary columns
    sampled_df = df.select([time_column] + sensor_columns).to_pandas()
    print(f"  Sampled data shape: {sampled_df.shape}")

    # Convert time column
    sampled_df[time_column] = pd.to_datetime(
        sampled_df[time_column], 
        format='%Y/%m/%d %H:%M:%S:%f', 
        errors="coerce", 
        exact=False
    )
    
    # Define time window
    start_time = pd.to_datetime(start_time)
    end_time = start_time + pd.Timedelta(minutes=duration_mins)

    # Filter to specific time frame
    sampled_df = sampled_df[
        (sampled_df[time_column] >= start_time) & 
        (sampled_df[time_column] <= end_time)
    ]
    
    print(f"  Filtered to {len(sampled_df)} samples in {duration_mins}-min window")
    return sampled_df


def _get_filtered_mask(sensor_series: pd.Series, threshold: float, sample_period: int) -> np.ndarray:
    """
    Return a boolean mask of the same length as sensor_series.
    True  → sample is inside an active window (|value| >= threshold,
            plus the sample_period tail after each crossing).
    False → sample is outside every active window.
    """
    n    = len(sensor_series)
    mask = np.zeros(n, dtype=bool)
    vals = sensor_series.to_numpy()

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


def find_sensor_peaks(
    df: pd.DataFrame,
    sensor_ids: list,
    time_column: str,
    threshold: float,
    sample_period: int
):
    """
    Find one first peak and one dominant peak per event window.
 
    Returns:
        Dict mapping sensor_id -> list of
        (first_peak_time, dominant_peak_time, first_peak_amplitude, dominant_peak_amplitude)
    """
    results = {}
 
    for sensor_id in sensor_ids:
        sensor_series = pd.Series(df[sensor_id].values, dtype=float)
        raw_signal    = sensor_series.to_numpy()
        time_series   = df[time_column]
        events        = []
 
        # Step 1: get event mask
        mask = _get_filtered_mask(sensor_series, threshold, sample_period)
 
        # Step 2: find contiguous True regions
        diff   = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0] + 1
 
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(mask))
 
        # Step 3: find extrema within each window
        for win_start, win_end in zip(starts, ends):
            seg = raw_signal[win_start:win_end]
            seg_len = len(seg)
 
            extrema = []
 
            # Boundary peak check
            if (seg[0] > 0 and seg[0] > seg[1]) or \
               (seg[0] < 0 and seg[0] < seg[1]):
                extrema.append(0)
 
            # Interior extrema
            for k in range(1, seg_len - 1):
                if seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1]:
                    extrema.append(k)
                elif seg[k] <= seg[k - 1] and seg[k] <= seg[k + 1]:
                    extrema.append(k)
 
            # Fallback if no extrema
            if not extrema:
                fallback = int(np.argmax(np.abs(seg)))
                extrema.append(fallback)
 
            # First peak
            first_peak_local = extrema[0]
 
            # Dominant peak
            dominant_local = max(extrema, key=lambda idx: abs(seg[idx]))
 
            # Convert to global indices
            fp_idx = win_start + first_peak_local
            dp_idx = win_start + dominant_local
 
            events.append((
                time_series.iloc[fp_idx],          # first_peak_time
                time_series.iloc[dp_idx],          # dominant_peak_time
                float(raw_signal[fp_idx]),         # first_peak_amplitude
                float(raw_signal[dp_idx]),         # dominant_peak_amplitude
            ))
 
        results[sensor_id] = events
        #print(f"    {sensor_id}: {len(events)} event(s)")
 
    return results


# =========================================================
# VEHICLE EVENT PROCESSING
# =========================================================
def process_vehicle_event(
    vehicle_id: str,
    timestamp_str: str,
    root_folder: str,
    boundary_sensors: list,
    junctions: list,
    threshold: float,
    sample_period: int,
    time_offset_minutes: int,
    duration_minutes: int
):
    """
    Process a single vehicle event.
    
    Args:
        vehicle_id: Vehicle identifier (e.g., "v1", "v2")
        timestamp_str: Original timestamp from weighing system (format: "YYYY-MM-DD HH:MM:SS")
        root_folder: Root data folder
        boundary_sensors: List of boundary sensor IDs
        junctions: List of BoundaryJunction objects from bridge model
        threshold: Peak detection threshold
        sample_period: Sample period for event windowing
        time_offset_minutes: Minutes to add to timestamp (weighing system delay)
        duration_minutes: Duration of analysis window
        
    Returns:
        Dictionary with sensor pair data
    """
    print(f"\n{'='*60}")
    print(f"Processing {vehicle_id}: {timestamp_str}")
    print(f"{'='*60}")
    
    # Step 1: Adjust timestamp
    original_time = pd.to_datetime(timestamp_str, dayfirst=TRUE )
    adjusted_time = original_time + timedelta(minutes=time_offset_minutes)
    
    print(f"Original timestamp: {original_time}")
    print(f"Adjusted timestamp: {adjusted_time} (+{time_offset_minutes} min)")
    
    # Step 2: Find CSV file
    date_str = adjusted_time.strftime("%Y%m%d")
    hour = adjusted_time.hour
    
    print(f"Looking for CSV: date={date_str}, hour={hour}")
    csv_path = find_csv_file(root_folder, date_str, hour)
    print(f"Found: {csv_path}")
    
    # Step 3: Load and filter data
    print(f"Loading CSV with Polars...")
    df_full = pl.read_csv(csv_path, separator=';')
    #print(df_full.head())



    # df_full['time'] = pd.to_datetime(df_full['time'], 
    # format='%Y/%d/%m %H:%M:%S:%f',  # Changed: Day/Month instead of Month/Day
    # errors="coerce", 
    # exact=False)
    adjusted_time = pd.to_datetime(adjusted_time, format= '%Y/%d/%m %H:%M:%S')


    print(f"Extracting {duration_minutes}-min window...")

    print(adjusted_time)


    df_window = get_only_interested_duration(
        df_full,
        sensor_columns=boundary_sensors,
        time_column='time',
        start_time=adjusted_time,
        duration_mins=duration_minutes
    )
    #print(df_window.head())
    # Remove DC offset (mean centering)
    for sensor in boundary_sensors:
        df_window[sensor] = df_window[sensor] - df_window[sensor].mean()
    
    # Step 4: Find peaks for all boundary sensors
    print(f"Detecting peaks (threshold={threshold}, sample_period={sample_period})...")
    peak_results = find_sensor_peaks(
        df_window,
        sensor_ids=boundary_sensors,
        time_column='time',
        threshold=threshold,
        sample_period=sample_period
    )
    
        # Step 5: Extract timestamps for all boundary sensors
    # Simple flat structure: each sensor gets 2 columns (first_peak, peak)
    result = {'vehicle_id': vehicle_id, 'original_timestamp': timestamp_str}
    
    for sensor_id in boundary_sensors:
        events = peak_results.get(sensor_id, [])
        
        if events:
            # Take first event from this sensor
            first_peak_time, peak_time, _, _ = events[0]
            result[f"{sensor_id}_first_peak"] = first_peak_time
            result[f"{sensor_id}_peak"] = peak_time
        else:
            # No events detected
            result[f"{sensor_id}_first_peak"] = None
            result[f"{sensor_id}_peak"] = None
            print(f"  WARNING: No events for sensor {sensor_id}")
    
    return result


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze vehicle events across bridge boundary sensors"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='CSV file with vehicle timestamps (single column: timestamp)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV path for results'
    )
    
    parser.add_argument(
        '--root_folder',
        type=str,
        required=True,
        help='Root folder containing date-organized sensor data'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.002,
        help='Peak detection threshold (default: 0.002)'
    )
    
    parser.add_argument(
        '--sample_period',
        type=int,
        default=500,
        help='Sample period for event windowing (default: 500)'
    )
    
    parser.add_argument(
        '--time_offset',
        type=int,
        default=1.5,
        help='Minutes to add to weighing system timestamp (default: 2)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Analysis window duration in minutes (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Load bridge model
    print("Loading bridge model...")
    bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    junctions = bridge.find_boundaries()
    
    # Get all boundary sensor IDs (in physical order)
    boundary_sensors = [s for j in junctions for s in j.sensor_ids()]
    # print(f"Boundary sensors ({len(boundary_sensors)}): {boundary_sensors}")
    # print(f"Junctions ({len(junctions)}):")
    # for j in junctions:
    #     print(f"  {j}: tail={j.sensor_ids()[:2]}, entry={j.sensor_ids()[2:]}")
    
    # Load vehicle timestamps
    print(f"\nLoading vehicle timestamps from: {args.input}")
    vehicles_df = pd.read_csv(args.input)
    
    if 'StartTimeStr' not in vehicles_df.columns:
        raise ValueError("Input CSV must have 'timestamp' column")

    
    # Remove duplicate timestamps
    print(f"Loaded {len(vehicles_df)} entries")
    vehicles_df = vehicles_df.drop_duplicates(subset=['StartTimeStr'], keep='first')
    print(f"After removing duplicate timestamps: {len(vehicles_df)} unique vehicle events")
    
    # Process each vehicle
    all_results = []
    
    for idx, row in vehicles_df.iterrows():
        vehicle_id = f"v{idx + 1}"  # Row index becomes vehicle ID (v1, v2, v3...)
        timestamp = row['StartTimeStr']
        
        try:
            result = process_vehicle_event(
                vehicle_id=vehicle_id,
                timestamp_str=timestamp,
                root_folder=args.root_folder,
                boundary_sensors=boundary_sensors,
                junctions=junctions,
                threshold=args.threshold,
                sample_period=args.sample_period,
                time_offset_minutes=args.time_offset,
                duration_minutes=args.duration
            )
            all_results.append(result)
            print(f"✓ {vehicle_id} processed successfully")
            
        except Exception as e:
            print(f"✗ ERROR processing {vehicle_id}: {e}")
            # Still add a row with NaN values
            result = {'vehicle_id': vehicle_id, 'original_timestamp': timestamp}
            all_results.append(result)
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results to: {args.output}")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output, index=False)
    print(f"Saved {len(results_df)} vehicle events")
    #print(f"Columns: {list(results_df.columns)}")
    print(f"{'='*60}")
    

if __name__ == "__main__":
    main()


    #ex: python3 cross_sensitivity_analysis.py --input timestamps_collection.csv --output res.csv --root_folder /Users/thomas/Data/Data_sensors