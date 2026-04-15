"""
Visualize Vehicle Event Analysis Results

Reads the output CSV from vehicle_event_analysis.py and creates:
1. Timeline plot - Shows when each vehicle was detected by each sensor
2. Sensor progression plot - Shows vehicle movement through sensors over time

Usage:
    python visualize_vehicle_events.py --input results.csv --output_dir plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
from pathlib import Path
import re


def extract_sensor_columns(df):
    """
    Extract sensor column information from the dataframe.
    
    Returns:
        sensor_list: Ordered list of unique sensor IDs
        first_peak_cols: Dict mapping sensor_id -> column name for first_peak
        peak_cols: Dict mapping sensor_id -> column name for dominant peak
    """
    # Find all columns ending with '_first_peak' or '_peak'
    first_peak_cols = {}
    peak_cols = {}
    sensor_order = []
    
    for col in df.columns:
        if col.endswith('_first_peak'):
            # Extract sensor ID (everything before '_first_peak')
            sensor_id = col.replace('_first_peak', '')
            first_peak_cols[sensor_id] = col
            if sensor_id not in sensor_order:
                sensor_order.append(sensor_id)
        elif col.endswith('_peak') and not col.endswith('_first_peak'):
            # Extract sensor ID (everything before '_peak')
            sensor_id = col.replace('_peak', '')
            peak_cols[sensor_id] = col
    
    print(f"Found {len(sensor_order)} unique sensors")
    return sensor_order, first_peak_cols, peak_cols


def plot_timeline(df, sensor_list, first_peak_cols, peak_cols, output_path):
    """
    Create timeline plot showing when each vehicle was detected by each sensor.
    
    X-axis: Absolute time (datetime)
    Y-axis: Vehicle events (v1, v2, v3...)
    Markers: Blue dots for first_peak, red dots for dominant_peak
    """
    print("\n" + "="*60)
    print("Generating Timeline Plot")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.5)))
    
    # Y-axis positions for each vehicle
    y_positions = {vehicle_id: idx for idx, vehicle_id in enumerate(df['vehicle_id'])}
    
    # Collect all timestamps for axis limits
    all_times = []
    
    # Plot each vehicle's detections
    for idx, row in df.iterrows():
        vehicle_id = row['vehicle_id']
        y_pos = y_positions[vehicle_id]
        
        # Plot first_peak detections (blue)
        for sensor_id in sensor_list:
            if sensor_id in first_peak_cols:
                col_name = first_peak_cols[sensor_id]
                timestamp = row[col_name]
                
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    all_times.append(time_obj)
                    ax.plot(time_obj, y_pos, 'o', color='blue', markersize=6, alpha=0.6)
        
        # Plot dominant_peak detections (red)
        for sensor_id in sensor_list:
            if sensor_id in peak_cols:
                col_name = peak_cols[sensor_id]
                timestamp = row[col_name]
                
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    all_times.append(time_obj)
                    ax.plot(time_obj, y_pos, 's', color='red', markersize=5, alpha=0.6)
    
    # Configure Y-axis
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['vehicle_id'])
    ax.set_ylabel('Vehicle Event', fontsize=12, fontweight='bold')
    
    # Configure X-axis
    if all_times:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Detection Timeline Across Sensors', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
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
    print(f"✓ Timeline plot saved to: {output_path}")
    plt.close()


def plot_sensor_progression(df, sensor_list, first_peak_cols, peak_cols, output_path):
    """
    Create sensor progression plot showing vehicle movement through sensors.
    
    X-axis: Sensor index (0, 1, 2, 3...)
    Y-axis: Time relative to first detection (seconds)
    Lines: Solid for first_peak, dashed for dominant_peak
    Different color per vehicle
    """
    print("\n" + "="*60)
    print("Generating Sensor Progression Plot")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color map for vehicles
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    
    # Plot each vehicle
    for idx, row in df.iterrows():
        vehicle_id = row['vehicle_id']
        color = colors[idx]
        
        # Collect first_peak times
        first_peak_times = []
        first_peak_indices = []
        
        for sensor_idx, sensor_id in enumerate(sensor_list):
            if sensor_id in first_peak_cols:
                col_name = first_peak_cols[sensor_id]
                timestamp = row[col_name]
                
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    first_peak_times.append(time_obj)
                    first_peak_indices.append(sensor_idx)
        
        # Collect dominant_peak times
        dominant_peak_times = []
        dominant_peak_indices = []
        
        for sensor_idx, sensor_id in enumerate(sensor_list):
            if sensor_id in peak_cols:
                col_name = peak_cols[sensor_id]
                timestamp = row[col_name]
                
                if pd.notna(timestamp):
                    time_obj = pd.to_datetime(timestamp)
                    dominant_peak_times.append(time_obj)
                    dominant_peak_indices.append(sensor_idx)
        
        # Convert to relative times (seconds from first detection)
        if first_peak_times:
            reference_time = min(first_peak_times)
            
            # First peak line (solid)
            relative_first = [(t - reference_time).total_seconds() for t in first_peak_times]
            ax.plot(first_peak_indices, relative_first, '-o', 
                   color=color, linewidth=2, markersize=6,
                   label=f'{vehicle_id} (first peak)', alpha=0.7)
            
            # Dominant peak line (dashed)
            if dominant_peak_times:
                relative_dominant = [(t - reference_time).total_seconds() for t in dominant_peak_times]
                ax.plot(dominant_peak_indices, relative_dominant, '--s', 
                       color=color, linewidth=1.5, markersize=5,
                       label=f'{vehicle_id} (dominant peak)', alpha=0.5)
        else:
            print(f"  WARNING: No detections for {vehicle_id}")
    
    # Configure X-axis
    ax.set_xticks(range(len(sensor_list)))
    ax.set_xticklabels(sensor_list, rotation=90, ha='right', fontsize=8)
    ax.set_xlabel('Sensor (Physical Order)', fontsize=12, fontweight='bold')
    
    # Configure Y-axis
    ax.set_ylabel('Time Since First Detection (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Vehicle Progression Through Sensors', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend (outside plot if many vehicles)
    if len(df) <= 10:
        ax.legend(loc='best', fontsize=9)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Sensor progression plot saved to: {output_path}")
    plt.close()


# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================
def extract_sensor_columns(df):
    """Extract sensor column information from results dataframe."""
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
    
    return sensor_order, first_peak_cols, peak_cols
 
 
def plot_timeline(df, sensor_list, first_peak_cols, peak_cols, output_path):
    """
    Timeline plot showing when each vehicle was detected by each sensor.
    
    Y-axis: Vehicles, X-axis: Time
    Blue dots = first_peak, Red squares = dominant_peak
    """
    print("\n  Generating timeline plot...")
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.5)))
    
    y_positions = {vehicle_id: idx for idx, vehicle_id in enumerate(df['vehicle_id'])}
    all_times = []
    
    # Plot detections
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
    
    # Configure axes
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
 
 
def plot_sensor_progression(df, sensor_list, first_peak_cols, peak_cols, output_path):
    """
    Sensor progression plot showing vehicle movement through sensors.
    
    X-axis: Sensor index, Y-axis: Time since first detection (seconds)
    Solid lines = first_peak, Dashed lines = dominant_peak
    """
    print("\n  Generating sensor progression plot...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    
    # Plot each vehicle
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
    
    # Configure axes
    ax.set_xticks(range(len(sensor_list)))
    ax.set_xticklabels(sensor_list, rotation=90, ha='right', fontsize=8)
    ax.set_xlabel('Sensor (Physical Order)', fontsize=12, fontweight='bold')
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
    print(f" Progression saved: {output_path}")
    plt.close()
 
 
def generate_visualizations(results_csv, output_dir):
    """Generate timeline and progression plots from results CSV."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Load results
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} vehicle events")
    
    # Extract sensor info
    sensor_list, first_peak_cols, peak_cols = extract_sensor_columns(df)
    print(f"Found {len(sensor_list)} sensors")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    timeline_path = output_dir / 'vehicle_timeline.png'
    plot_timeline(df, sensor_list, first_peak_cols, peak_cols, timeline_path)
    
    progression_path = output_dir / 'vehicle_progression.png'
    plot_sensor_progression(df, sensor_list, first_peak_cols, peak_cols, progression_path)
    
    print("\n" + "="*60)
    print("Visualizations complete!")
    print("="*60)
 



def main():

    parser = argparse.ArgumentParser( description="Visualize vehicle event analysis results")
    
    parser.add_argument('--input', type=str, required=True, help='Input CSV file from vehicle_event_analysis.py')
    
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots (default: current directory)')
    
    parser.add_argument('--timeline', action='store_true', help='Generate timeline plot')
    
    parser.add_argument('--progression', action='store_true', help='Generate sensor progression plot')
    
    args = parser.parse_args()
    
    # If no plot flags specified, generate both
    if not args.timeline and not args.progression:
        args.timeline = True
        args.progression = True
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Vehicle Event Visualization")
    print("="*60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} vehicle events")
    print(f"Total columns: {len(df.columns)}")
    
    # Extract sensor information
    sensor_list, first_peak_cols, peak_cols = extract_sensor_columns(df)
    
    # Generate plots
    if args.timeline:
        timeline_path = output_dir / 'vehicle_timeline.png'
        plot_timeline(df, sensor_list, first_peak_cols, peak_cols, timeline_path)
    
    if args.progression:
        progression_path = output_dir / 'vehicle_progression.png'
        plot_sensor_progression(df, sensor_list, first_peak_cols, peak_cols, progression_path)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()