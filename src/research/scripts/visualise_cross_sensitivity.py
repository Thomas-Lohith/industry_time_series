"""
Visualize Vehicle Event Analysis Results
Adapted for new dominant_peaks format: "timestamp|amplitude;timestamp|amplitude"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter


def parse_peaks_cell(cell):
    """
    Parse "ts1|val1;ts2|val2;..." into list of (timestamp, amplitude).
    Returns [] if cell is NaN or empty.
    """
    if pd.isna(cell) or str(cell).strip() == '':
        return []
    peaks = []
    for entry in str(cell).split(';'):
        parts = entry.strip().split('|')
        if len(parts) == 2:
            try:
                ts  = pd.to_datetime(parts[0].strip())
                amp = float(parts[1].strip())
                peaks.append((ts, amp))
            except Exception:
                continue
    return peaks


def extract_sensor_columns(df, boundary_sensors):
    """
    Find columns matching pattern: {sensor_id}_dominant_peaks
    Returns ordered list of sensor_ids found in the CSV.
    """
    sensor_order = []
    for sensor_id in boundary_sensors:
        col = f"{sensor_id}_dominant_peaks"
        if col in df.columns:
            sensor_order.append(sensor_id)
        else:
            print(f"  WARNING: Column '{col}' not found in CSV")
    return sensor_order


def plot_sensor_progression(df, sensor_list, sensor_numbers, output_path):
    print("\n  Generating sensor progression plot...")

    MAX_COMBOS = 200  # safety cap per vehicle

    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    legend_handles = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        vehicle_id = row['vehicle_id']
        color = colors[row_idx]

        # Parse all peaks per sensor
        sensor_peaks = {}
        for s_idx, sensor_id in enumerate(sensor_list):
            col = f"{sensor_id}_dominant_peaks"
            peaks = parse_peaks_cell(row.get(col, np.nan))
            if peaks:
                sensor_peaks[s_idx] = peaks

        if not sensor_peaks:
            print(f"  WARNING: No peaks at all for {vehicle_id}")
            continue

        # Reference time = earliest timestamp across all sensors
        all_timestamps = [ts for peaks in sensor_peaks.values() for ts, _ in peaks]
        reference_time = min(all_timestamps)

        sensors_with_data = sorted(sensor_peaks.keys())
        options_per_sensor = [
            [(s_idx, (ts - reference_time).total_seconds()) for ts, _ in sensor_peaks[s_idx]]
            for s_idx in sensors_with_data
        ]

        # --- FIX 1+2: generator + cap, FIX 3: LineCollection ---
        segments = []
        for i, combo in enumerate(itertools.product(*options_per_sensor)):  # FIX 1: no list()
            if i >= MAX_COMBOS:                                              # FIX 2: hard cap
                print(f"  WARNING: {vehicle_id} hit combo cap ({MAX_COMBOS}), truncating")
                break
            segments.append([(pt[0], pt[1]) for pt in combo])

        if not segments:
            continue

        # FIX 3: one LineCollection per vehicle instead of N ax.plot() calls
        lc = LineCollection(
            segments,
            colors=[color],
            linewidths=1.2,
            alpha=0.45
        )
        ax.add_collection(lc)

        # Also plot marker dots separately (LineCollection has no markers)
        all_xs = [x for seg in segments for x, _ in seg]
        all_ys = [y for seg in segments for _, y in seg]
        ax.scatter(all_xs, all_ys, color=color, s=12, alpha=0.5, zorder=3)

        # Update axis limits manually (LineCollection doesn't auto-scale)
        ax.update_datalim(np.column_stack([all_xs, all_ys]))
        ax.autoscale_view()

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2, label=vehicle_id)
        )

    # X-axis sensor labels
    ax.set_xticks(range(len(sensor_list)))
    x_labels = [f"{sid}\n({sensor_numbers.get(sid, '?')})" for sid in sensor_list]
    ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=7)
    ax.set_xlabel('Sensor ID (Sensor Number)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time Since First Detection (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Vehicle Possible Tracks Through Sensors\n(all dominant peak combinations per window)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(True, alpha=0.3, linestyle='--')

    if len(df) <= 12:
        ax.legend(handles=legend_handles, loc='best', fontsize=9)
    else:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1),
                  loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Progression saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize vehicle event analysis results")
    parser.add_argument('--input',       type=str, required=True,  help='Input CSV from vehicle_event_analysis.py')
    parser.add_argument('--output_dir',  type=str, default='.',    help='Output directory for plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Vehicle Event Visualization")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {output_dir}")

    # Load bridge model
    print("\nLoading bridge model...")
    bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    junctions = bridge.find_boundaries()
    boundary_sensors = list(dict.fromkeys(s for j in junctions for s in j.sensor_ids()))
    boundary_nums    = list(dict.fromkeys(s for j in junctions for s in j.sensor_numbers()))
    sensor_numbers   = dict(zip(boundary_sensors, boundary_nums))
    print(f"Loaded {len(boundary_sensors)} boundary sensors")

    # Load CSV
    print("\nLoading data...")
    df = pd.read_csv(args.input)
    df = df[:1]
    print(f"Loaded {len(df)} vehicle events")

    sensor_list = extract_sensor_columns(df, boundary_sensors)
    print(f"Found {len(sensor_list)} sensors with data columns")

    plot_sensor_progression(df, sensor_list, sensor_numbers,
                            output_dir / 'vehicle_progression_trial.png')

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    # ex: python3 visualize_vehicle_events.py --input res.csv --output_dir plots/