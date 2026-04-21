#!/usr/bin/env python3
"""
Visualize Vehicle Event Analysis Results

Standalone visualization script that reads output CSV from vehicle_event_analysis.py
and creates timeline and sensor progression plots.

Usage:
    python visualize_vehicle_events.py --input results.csv --output_dir plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from pathlib import Path

# Import bridge model to get sensor numbers
from src.shared.bridge_model import load_bridge
from src.shared.config import position_csv, threshold_csv, delimiter


def extract_sensor_columns(df, boundary_sensors):
    """
    Extract and validate sensor columns from results dataframe.
    
    Returns:
        sensor_list, first_peak_cols, peak_cols
    """
    first_peak_cols = {}
    peak_cols = {}
    sensor_order = []
    
    for col in df.columns:
        if col.endswith('_first_peak'):
            sensor_id = col.replace('_first_peak', '')
            first_peak_cols[sensor_id] = col
            if sensor_id not in sensor_order:
                sensor_order.append(sensor_id)
        elif col.endswith('_peak') and not col.endswith('_first_peak'):
            sensor_id = col.replace('_peak', '')
            peak_cols[sensor_id] = col
    
    # Validate against boundary sensors
    expected_set = set(boundary_sensors)
    found_set = set(sensor_order)
    
    if found_set != expected_set:
        missing = expected_set - found_set
        extra = found_set - expected_set
        if missing:
            print(f"  WARNING: Missing in CSV: {missing}")
        if extra:
            print(f"  WARNING: Extra in CSV: {extra}")
    
    return sensor_order, first_peak_cols, peak_cols


def plot_timeline(df, sensor_list, first_peak_cols, peak_cols, output_path):
    """Timeline plot: Y=vehicles, X=time, Blue=first_peak, Red=dominant_peak"""
    print("\n  Generating timeline plot...")
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.5)))
    y_positions = {vehicle_id: idx for idx, vehicle_id in enumerate(df['vehicle_id'])}
    all_times = []
    
    for idx, row in df.iterrows():
        vehicle_id = row['vehicle_id']
        y_pos = y_positions[vehicle_id]
        
        # First peaks (blue)
        for sensor_id in sensor_list:
            if sensor_id in first_peak_cols:
                timestamp = row[first_peak_cols[sensor_id]]
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    all_times.append(time_obj)
                    ax.plot(time_obj, y_pos, 'o', color='blue', markersize=6, alpha=0.6)
        
        # Dominant peaks (red)
        for sensor_id in sensor_list:
            if sensor_id in peak_cols:
                timestamp = row[peak_cols[sensor_id]]
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    all_times.append(time_obj)
                    ax.plot(time_obj, y_pos, 's', color='red', markersize=5, alpha=0.6)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['vehicle_id'])
    ax.set_ylabel('Vehicle Event', fontsize=12, fontweight='bold')
    
    if all_times:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Detection Timeline', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='First Peak', alpha=0.6),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=7, label='Dominant Peak', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Timeline saved: {output_path}")
    plt.close()


def plot_sensor_progression(df, sensor_list, first_peak_cols, peak_cols, sensor_numbers, output_path):
    """Progression plot: X=sensor index, Y=time since first detection"""
    print("\n  Generating sensor progression plot...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    
    for idx, row in df.iterrows():
        vehicle_id = row['vehicle_id']
        color = colors[idx]
        
        # Collect first_peak times
        first_peak_times = []
        first_peak_indices = []
        
        for sensor_idx, sensor_id in enumerate(sensor_list):
            if sensor_id in first_peak_cols:
                timestamp = row[first_peak_cols[sensor_id]]
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    first_peak_times.append(time_obj)
                    first_peak_indices.append(sensor_idx)
        
        # Collect dominant_peak times
        dominant_peak_times = []
        dominant_peak_indices = []
        
        for sensor_idx, sensor_id in enumerate(sensor_list):
            if sensor_id in peak_cols:
                timestamp = row[peak_cols[sensor_id]]
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    dominant_peak_times.append(time_obj)
                    dominant_peak_indices.append(sensor_idx)
        
        # Convert to relative times
        if first_peak_times:
            reference_time = min(first_peak_times)
            
            # First peak line (solid)
            relative_first = [(t - reference_time).total_seconds() for t in first_peak_times]
            ax.plot(first_peak_indices, relative_first, '-o', 
                   color=color, linewidth=2, markersize=6,
                   label=f'{vehicle_id} (first)', alpha=0.7)
            
            # Dominant peak line (dashed)
            if dominant_peak_times:
                relative_dominant = [(t - reference_time).total_seconds() for t in dominant_peak_times]
                ax.plot(dominant_peak_indices, relative_dominant, '--s', 
                       color=color, linewidth=1.5, markersize=5,
                       label=f'{vehicle_id} (dominant)', alpha=0.5)
    
    # X-axis labels: sensor_id (sensor_num)
    ax.set_xticks(range(len(sensor_list)))
    x_labels = []
    for sensor_id in sensor_list:
        sensor_num = sensor_numbers.get(sensor_id, '?')
        x_labels.append(f"{sensor_id}-({sensor_num})")
    ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=8)
    ax.set_xlabel('Sensor ID (Sensor Number)', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Time Since First Detection (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Progression Through Sensors', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    if len(df) <= 10:
        ax.legend(loc='best', fontsize=9)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Progression saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize vehicle event analysis results")
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV from vehicle_event_analysis.py')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots')
    parser.add_argument('--timeline', action='store_true',
                       help='Generate timeline plot only')
    parser.add_argument('--progression', action='store_true',
                       help='Generate progression plot only')
    
    args = parser.parse_args()
    
    # If no plot flags, generate both
    if not args.timeline and not args.progression:
        args.timeline = False
        args.progression = True
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Vehicle Event Visualization")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    
    # Load bridge model for sensor numbers
    print("\nLoading bridge model...")
    bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    junctions = bridge.find_boundaries()
    
    boundary_sensors = [s for j in junctions for s in j.sensor_ids()]
    boundary_nums = [s for j in junctions for s in j.sensor_numbers()]
    
    boundary_sensors = list(dict.fromkeys(boundary_sensors))
    boundary_nums = list(dict.fromkeys(boundary_nums))
    
    sensor_numbers = dict(zip(boundary_sensors, boundary_nums))
    print(f"Loaded {len(boundary_sensors)} boundary sensors")
    
    # Load results
    print("\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} vehicle events")
    
    # Extract and validate sensors
    sensor_list, first_peak_cols, peak_cols = extract_sensor_columns(df, boundary_sensors)
    print(f"Found {len(sensor_list)} sensors in results")
    
    # Generate plots
    if args.timeline:
        plot_timeline(df, sensor_list, first_peak_cols, peak_cols, 
                     output_dir / 'vehicle_timeline.png')
    
    if args.progression:
        plot_sensor_progression(df, sensor_list, first_peak_cols, peak_cols, 
                               sensor_numbers, output_dir / 'vehicle_progression_2.png')
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()