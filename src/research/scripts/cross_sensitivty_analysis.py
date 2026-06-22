"""Vehicle Event Analysis: Extract peak timestamps from boundary sensor pairs"""

from pickle import TRUE
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import os
import argparse
from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter


def find_csv_file(root_folder, date_str, hour):
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    csv_folder = os.path.join(root_folder, date_str, date_str, "csv_acc")
    for fname in os.listdir(csv_folder):
        if (fname.startswith(f"M001_{formatted_date}_{hour:02d}-00-00_")
                and f"_int-{hour+1}_" in fname and fname.endswith("_th.csv")):
            return os.path.join(csv_folder, fname)
    raise FileNotFoundError(f"No matching CSV file found for {date_str} hour {hour}")


def get_only_interested_duration(df, sensor_columns, time_column, start_time, duration_mins):
    sampled_df = df.select([time_column] + sensor_columns).to_pandas()
    sampled_df[time_column] = pd.to_datetime(sampled_df[time_column], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    start_time = pd.to_datetime(start_time)
    end_time = start_time + pd.Timedelta(minutes=duration_mins)
    return sampled_df[(sampled_df[time_column] >= start_time) & (sampled_df[time_column] <= end_time)]


def _get_filtered_mask(sensor_series, threshold, sample_period):
    n, mask, vals = len(sensor_series), np.zeros(len(sensor_series), dtype=bool), sensor_series.to_numpy()
    i = 0
    while i < n:
        if np.abs(vals[i]) >= threshold:
            start, end = i, min(i + sample_period, n)
            while i < end:
                if np.abs(vals[i]) >= threshold:
                    end = min(i + sample_period, n)
                i += 1
            mask[start:end] = True
        else:
            i += 1
    return mask


def find_sensor_peaks(df, sensor_ids, time_column, threshold, sample_period):
    results = {}
    for sensor_id in sensor_ids:
        sensor_series = pd.Series(df[sensor_id].values, dtype=float)
        raw_signal, time_series, events = sensor_series.to_numpy(), df[time_column], []
        mask = _get_filtered_mask(sensor_series, threshold, sample_period)
        diff = np.diff(mask.astype(int))
        starts, ends = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0] + 1
        if mask[0]: starts = np.insert(starts, 0, 0)
        if mask[-1]: ends = np.append(ends, len(mask))

        for win_start, win_end in zip(starts, ends):
            seg = raw_signal[win_start:win_end]
            extrema = ([0] if (seg[0] > 0 and seg[0] > seg[1]) or (seg[0] < 0 and seg[0] < seg[1]) else [])
            extrema += [k for k in range(1, len(seg) - 1) if (seg[k] >= seg[k-1] and seg[k] >= seg[k+1]) or (seg[k] <= seg[k-1] and seg[k] <= seg[k+1])]
            if not extrema: extrema.append(int(np.argmax(np.abs(seg))))
            fp_idx = win_start + extrema[0]
            dp_idx = win_start + max(extrema, key=lambda idx: abs(seg[idx]))
            events.append((time_series.iloc[fp_idx], time_series.iloc[dp_idx], float(raw_signal[fp_idx]), float(raw_signal[dp_idx])))

        results[sensor_id] = events
    return results


def process_vehicle_event(vehicle_id, timestamp_str, root_folder, boundary_sensors, junctions, threshold, sample_period, time_offset_minutes, duration_minutes):
    print(f"\n{'='*60}\nProcessing {vehicle_id}: {timestamp_str}\n{'='*60}")

    original_time = pd.to_datetime(timestamp_str, dayfirst=TRUE)
    print(f"Original: {original_time}")
    adjusted_time = original_time + timedelta(minutes=time_offset_minutes)
    print(f"Original: {original_time} → Adjusted: {adjusted_time} (+{time_offset_minutes} min)")

    csv_path = find_csv_file(root_folder, adjusted_time.strftime("%Y%m%d"), adjusted_time.hour)
    print(f"Found: {csv_path}")

    df_full = pl.read_csv(csv_path, separator=';')
    adjusted_time = pd.to_datetime(adjusted_time, format='%Y/%m/%d %H:%M:%S')
    print(f"Adjusted: {adjusted_time}")

    df_window = get_only_interested_duration(df_full, boundary_sensors, 'time', adjusted_time, duration_minutes)
    for sensor in boundary_sensors:
        df_window[sensor] = df_window[sensor] - df_window[sensor].mean()

    peak_results = find_sensor_peaks(df_window, boundary_sensors, 'time', threshold, sample_period)

    result = {'vehicle_id': vehicle_id, 'original_timestamp': timestamp_str}
    for sensor_id in boundary_sensors:
        events = peak_results.get(sensor_id, [])
        if events:
            first_peak_time, peak_time, _, _ = events[0]
            result[f"{sensor_id}_first_peak"] = first_peak_time
            result[f"{sensor_id}_peak"] = peak_time
        else:
            result[f"{sensor_id}_first_peak"] = result[f"{sensor_id}_peak"] = None
            print(f"  WARNING: No events for sensor {sensor_id}")
    return result


def main():
    p = argparse.ArgumentParser(description="Analyze vehicle events across bridge boundary sensors")
    p.add_argument('--input',       required=True,              help='CSV with vehicle timestamps')
    p.add_argument('--output',      required=True,              help='Output CSV path')
    p.add_argument('--root_folder', required=True,              help='Root folder with sensor data')
    p.add_argument('--threshold',   type=float, default=0.002,  help='Peak detection threshold')
    p.add_argument('--sample_period', type=int, default=500,    help='Sample period for windowing')
    p.add_argument('--time_offset', type=int,   default=1.5,    help='Timestamp offset in minutes')
    p.add_argument('--duration',    type=int,   default=5,      help='Analysis window in minutes')
    args = p.parse_args()

    bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    junctions = bridge.find_boundaries()
    boundary_sensors = [s for j in junctions for s in j.sensor_ids()]

    vehicles_df = pd.read_csv(args.input)
    if 'StartTimeStr' not in vehicles_df.columns:
        raise ValueError("Input CSV must have 'StartTimeStr' column")

    vehicles_df = (vehicles_df.sort_values('GrossWeight', ascending=False).drop_duplicates(subset=['StartTimeStr'], keep='first'))
    print(f"Processing {len(vehicles_df)} unique vehicle events")

    all_results = []
    for idx, row in vehicles_df.iterrows():
        vehicle_id = f"v{idx + 1}"
        try:
            result = process_vehicle_event(vehicle_id, row['StartTimeStr'], args.root_folder,
                                           boundary_sensors, junctions, args.threshold,
                                           args.sample_period, args.time_offset, args.duration)
            print(f"✓ {vehicle_id} processed successfully")
        except Exception as e:
            print(f"✗ ERROR processing {vehicle_id}: {e}")
            result = {'vehicle_id': vehicle_id, 'original_timestamp': row['StartTimeStr']}
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(results_df)} vehicle events to {args.output}")


if __name__ == "__main__":
    main()
    #ex: python3 cross_sensitivity_analysis.py --input timestamps_collection.csv --output res.csv --root_folder /Users/thomas/Data/Data_sensors
    #python3 cross_sensitivty_analysis.py --input ../csv/timestamps_18_03_2025_30-45min.csv --output ../csv/res_18_03_2025_30-45min.csv --root_folder /home/thomas/backup-sensor-data