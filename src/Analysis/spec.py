import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime, timedelta
import glob
import os
import argparse
import sys

def find_csv_file(root_folder, date_str, hour):
    hour1 = hour + 1
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    csv_folder = os.path.join(root_folder, date_str, "csv_acc")

    for fname in os.listdir(csv_folder):
        if (
            fname.startswith(f"M001_{formatted_date}_{hour:02d}-00-00_")
            and f"_int-{hour1}_" in fname
            and fname.endswith("_th.csv")
        ):
            return os.path.join(csv_folder, fname)

    raise FileNotFoundError("No matching CSV file found.")


def extract_15min_data(csv_path, sensor_id, start_time_str):

    # Read the CSV file with semicolon separator
    df = pd.read_csv(csv_path, sep=';')
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    
    # Parse start time and calculate end time (15 minutes later)
    start_hour, start_min, start_sec = map(int, start_time_str.split(':'))
    
    # Get the date from the first row
    base_date = df['time'].iloc[0].date()
    start_datetime = datetime.combine(base_date, datetime.min.time().replace(
        hour=start_hour, minute=start_min, second=start_sec))
    end_datetime = start_datetime + timedelta(minutes=15)
    
    # Filter data for the 15-minute interval
    mask = (df['time'] >= start_datetime) & (df['time'] < end_datetime)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        raise ValueError(f"No data found for time range {start_datetime} to {end_datetime}")
    
    # Extract sensor data
    if sensor_id not in filtered_df.columns:
        raise ValueError(f"Sensor ID '{sensor_id}' not found in columns: {filtered_df.columns.tolist()}")
    
    data = filtered_df[sensor_id].values
    
    return data, start_datetime, end_datetime


def get_vehicle_counts(excel_path, date_str, start_time_str):
   
    # Read the Excel file (or CSV if that's the format)
    # Adjust the read function based on actual file format
    if excel_path.endswith('.xlsx') or excel_path.endswith('.xls'):
        df = pd.read_excel(excel_path)
    else:
        df = pd.read_csv(excel_path)
    
    # Convert date_str to match the format in the data
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    # Parse the time to find the matching row
    start_hour, start_min, _ = map(int, start_time_str.split(':'))
    
    # Create a time range string to match (e.g., "0:00 - 0:15")
    end_min = start_min + 15
    end_hour = start_hour
    if end_min >= 60:
        end_min -= 60
        end_hour += 1
    
    time_range = f"{start_hour}:{start_min:02d} - {end_hour}:{end_min:02d}"
    
    # Find matching row
    # The date column might have different formats, adjust as needed
    matching_row = None
    for idx, row in df.iterrows():
        if isinstance(row['Data e ora'], str):
            if time_range in row['Data e ora'] and day in row['Data e ora']:
                matching_row = row
                break
    
    if matching_row is None:
        return {"Car": 0, "Bus": 0, "Motorbike": 0, "Truck": 0, "Van": 0}
    
    # Extract vehicle counts
    vehicle_counts = {
        "Car": int(matching_row.get('Car', 0)),
        "Bus": int(matching_row.get('Bus', 0)),
        "Motorbike": int(matching_row.get('Motorbike', 0)),
        "Truck": int(matching_row.get('Truck', 0)),
        "Van": int(matching_row.get('Van', 0))
    }
    
    print(f"Found vehicle counts: {vehicle_counts}")
    return vehicle_counts


def compute_power_spectrum(data, sampling_rate):
   
    # Remove mean
    data = data - np.mean(data)
    
    # Compute power spectrum using Welch's method
    freqs, power = signal.welch(
        data,
        fs=sampling_rate,
        window='hann',
        nperseg=2048,
        scaling='density'
    )
    
    # Limit frequencies to 20 Hz
    freq_limit = 20
    mask = freqs <= freq_limit
    freqs, power = freqs[mask], power[mask]
    
    # Convert to µ-units (optional)
    power *= 1e6
    
    return freqs, power


def plot_spectrum_with_labels(freqs, power, vehicle_counts, start_time, end_time, sensor_id):

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot power spectrum
    ax.semilogy(freqs, power, 'b-', linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Spectrum', fontsize=12, fontweight='bold')
    ax.set_title(f'Power Spectrum - Sensor: {sensor_id}\n'
                 f'Time: {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%H:%M:%S")}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add vehicle count labels below the graph
    total_vehicles = sum(vehicle_counts.values())
    label_text = f"Vehicle Counts (15-min interval):\n"
    label_text += f"Total: {total_vehicles} | "
    label_text += " | ".join([f"{vtype}: {count}" for vtype, count in vehicle_counts.items()])
    
    # Add text box with vehicle counts
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.5, -0.15, label_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='center',
            bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig


def plot_combined_subplots(data_list, sensor_id, date_str, hour):

    n_plots = len(data_list)
    
    # Determine subplot layout (rows x cols)
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots == 2:
        rows, cols = 1, 2
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    elif n_plots <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16*cols, 8*rows))
    
    # Flatten axes array for easier iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each time interval
    for idx, (freqs, power, vehicle_counts, start_time, end_time) in enumerate(data_list):
        ax = axes[idx]
        
        # Plot power spectrum
        ax.semilogy(freqs, power, 'b-', linewidth=0.8)
        ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Power Spectrum', fontsize=10, fontweight='bold')
        ax.set_title(f'Time: {start_time.strftime("%H:%M:%S")} to {end_time.strftime("%H:%M:%S")}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add vehicle count labels
        total_vehicles = sum(vehicle_counts.values())
        label_text = f"Total: {total_vehicles}\n"
        label_text += " | ".join([f"{vtype}: {count}" for vtype, count in vehicle_counts.items()])
        
        # Add text box with vehicle counts
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.5, -0.11, label_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='top', horizontalalignment='center',
                bbox=props)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Add main title
    fig.suptitle(f'Power Spectrum Analysis - Sensor: {sensor_id} | Date: {date_str} | Hour: {hour:02d}:00',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.05, hspace=0.35, wspace=0.1)
    
    return fig


def plot_superimposed(data_list, sensor_id, date_str, hour):
 
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define color palette for different time intervals
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    # Plot each time interval with different color
    legend_labels = []
    total_vehicles_all = []
    
    for idx, (freqs, power, vehicle_counts, start_time, end_time) in enumerate(data_list):
        color = colors[idx]
        
        # Plot power spectrum
        ax.semilogy(freqs, power, linewidth=1.5, color=color, alpha=0.8)
        
        # Create legend label with time and total vehicles
        total_vehicles = sum(vehicle_counts.values())
        total_vehicles_all.append((start_time, end_time, total_vehicles, vehicle_counts))
        legend_labels.append(f'{start_time.strftime("%H:%M:%S")} - {end_time.strftime("%H:%M:%S")} (Total: {total_vehicles})')
    
    # Set labels and title
    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power Spectrum', fontsize=14, fontweight='bold')
    
    # Format date for display
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    formatted_date = f"{year}-{month}-{day}"
    
    ax.set_title(f'Superimposed Power Spectrum Analysis\nSensor: {sensor_id} | Date: {formatted_date} | Hour: {hour:02d}:00',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(legend_labels, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Create detailed vehicle count table below the plot
    table_text = "VEHICLE COUNTS BY TIME INTERVAL\n"
    table_text += "═" * 100 + "\n"
    
    for start_time, end_time, total, vehicle_counts in total_vehicles_all:
        time_str = f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"
        table_text += f"{time_str:25} | Total: {total:3} | "
        counts_str = " | ".join([f"{vtype}: {count:2}" for vtype, count in vehicle_counts.items()])
        table_text += counts_str + "\n"
    
    # Add text box with all vehicle counts
    props = dict(boxstyle='round,pad=0.3', facecolor='#fffacd', 
                edgecolor='#d4a017', linewidth=2, alpha=0.95)
    ax.text(0.5, -0.12, table_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='center',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    return fig


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Plot power spectrum of accelerometer data with vehicle count labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single time interval
  python script.py -r /data/root -e /data/traffic.xlsx -d 20250208 -H 0 -s 03091002_x -t 00:00:00
  
  # Multiple time intervals (separate plots)
  python script.py -r /data/root -e /data/traffic.xlsx -d 20250208 -H 0 -s 03091002_x -t 00:00:00 -t 00:15:00 -t 00:30:00
  
  # Multiple time intervals (combined as subplots)
  python script.py -r /data/root -e /data/traffic.xlsx -d 20250208 -H 0 -s 03091002_x -t 00:00:00 -t 00:15:00 -t 00:30:00 -t 00:45:00 --subplot
  
  # Multiple time intervals (superimposed on single plot)
  python script.py -r /data/root -e /data/traffic.xlsx -d 20250208 -H 0 -s 03091002_x -t 00:00:00 -t 00:15:00 -t 00:30:00 -t 00:45:00 --superimpose
        '''
    )
    
    # Required arguments
    parser.add_argument('-r', '--root-folder',
                        required=True,
                        help='Root directory containing date folders')
    
    parser.add_argument('-e', '--excel-path',
                        required=True,
                        help='Path to Excel/CSV file with vehicle counts')
    
    parser.add_argument('-d', '--date',
                        required=True,
                        help='Date in YYYYMMDD format, e.g., 20250208')
    
    parser.add_argument('-H', '--hour',
                        required=True,
                        type=int,
                        choices=range(0, 24),
                        metavar='HOUR',
                        help='Hour (0-23)')
    
    parser.add_argument('-s', '--sensor-id',
                        required=True,
                        help='Sensor column name, e.g., 03091002_x')
    
    parser.add_argument('-t', '--start-time',
                        required=True,
                        action='append',
                        help='Start time in HH:MM:SS format (can specify multiple times for batch processing)')
    
    # Optional arguments for plot mode
    plot_mode = parser.add_mutually_exclusive_group()
    plot_mode.add_argument('--subplot',
                          action='store_true',
                          help='Combine multiple time intervals as subplots in a single figure')
    
    plot_mode.add_argument('--superimpose',
                          action='store_true',
                          help='Overlay all time intervals on a single plot with different colors')
    
    return parser.parse_args()


def main_parametric(args):
    """
    Main function using parsed arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    # Hardcoded settings
    SAMPLING_RATE = 100  # Hz (100 samples per second)
    DPI = 300
    OUTPUT_DIR = '/home/c8x-98x/industry_time_series/src/results/filtering'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # SUPERIMPOSE MODE: Overlay all plots on single graph
    if args.superimpose and len(args.start_time) > 1:
        print(f"\n{'='*60}")
        print(f"SUPERIMPOSE MODE: Processing {len(args.start_time)} intervals \n")
        print(f"{'='*60}")
        
        data_list = []
        
        # Process each start time and collect data
        for idx, start_time_str in enumerate(args.start_time):
            print(f"\nProcessing interval {idx+1}/{len(args.start_time)}: {start_time_str}")
            
            try:
                # Find and read CSV file
                csv_path = find_csv_file(args.root_folder, args.date, args.hour)
                
                # Extract 15-minute data
                data, start_time, end_time = extract_15min_data(csv_path, args.sensor_id, start_time_str)
                print(f"  ** Extracted {len(data)} samples")
                
                # Get vehicle counts
                vehicle_counts = get_vehicle_counts(args.excel_path, args.date, start_time_str)
                
                # Compute power spectrum
                freqs, power = compute_power_spectrum(data, SAMPLING_RATE)
                
                # Store data for superimposed plot
                data_list.append((freqs, power, vehicle_counts, start_time, end_time))
                
            except Exception as e:
                print(f"  ** Error processing {start_time_str}: {str(e)}")
                continue
        
        # Create superimposed plot
        if data_list:
            print(f"\nCreating superimposed plot with {len(data_list)} curves...")
            fig = plot_superimposed(data_list, args.sensor_id, args.date, args.hour)
            
            # Save figure
            output_filename = f"spectrum_superimposed_{args.date}_{args.hour:02d}_{args.sensor_id}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
            print(f"** Saved superimposed plot to: {output_path}")
            
            # Display plot
            plt.show()
        else:
            print("** No data collected for superimposed plot")
    
    # SUBPLOT MODE: Collect all data first, then create single figure
    elif args.subplot and len(args.start_time) > 1:
        print(f"\n{'='*60}")
        print(f"SUBPLOT MODE: Processing {len(args.start_time)} intervals")
        print(f"{'='*60}")
        
        data_list = []
        
        # Process each start time and collect data
        for idx, start_time_str in enumerate(args.start_time):
            print(f"\nProcessing interval {idx+1}/{len(args.start_time)}: {start_time_str}")
            
            try:
                # Find and read CSV file
                csv_path = find_csv_file(args.root_folder, args.date, args.hour)
                
                # Extract 15-minute data
                data, start_time, end_time = extract_15min_data(csv_path, args.sensor_id, start_time_str)
                print(f"  ** Extracted {len(data)} samples")
                
                # Get vehicle counts
                vehicle_counts = get_vehicle_counts(args.excel_path, args.date, start_time_str)
                
                # Compute power spectrum
                freqs, power = compute_power_spectrum(data, SAMPLING_RATE)
                
                # Store data for subplot
                data_list.append((freqs, power, vehicle_counts, start_time, end_time))
                
            except Exception as e:
                print(f"  ** Error processing {start_time_str}: {str(e)}")
                continue
        
        # Create combined subplot figure
        if data_list:
            print(f"\nCreating combined subplot figure with {len(data_list)} plots...")
            fig = plot_combined_subplots(data_list, args.sensor_id, args.date, args.hour)
            
            # Save figure
            output_filename = f"spectrum_combined_{args.date}_{args.hour:02d}_{args.sensor_id}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
            print(f"** Saved combined plot to: {output_path}")
            
            # Display plot
            plt.show()
        else:
            print("** No data collected for subplots")
    
    # SEPARATE MODE: Create individual plots for each time interval
    else:
        if (args.subplot or args.superimpose) and len(args.start_time) == 1:
            print("Note: --subplot/--superimpose flag ignored for single time interval")
        
        # Process each start time separately
        for idx, start_time_str in enumerate(args.start_time):
            print(f"\n{'='*60}")
            print(f"Processing interval {idx+1}/{len(args.start_time)}")
            print(f"{'='*60}")
            
            try:
                # Find and read CSV file
                print(f"Searching for CSV file...")
                csv_path = find_csv_file(args.root_folder, args.date, args.hour)
                print(f"Found: {csv_path}")
                
                # Extract 15-minute data
                print(f"Extracting 15-minute data for sensor {args.sensor_id}...")
                data, start_time, end_time = extract_15min_data(csv_path, args.sensor_id, start_time_str)
                print(f"Extracted {len(data)} samples from {start_time} to {end_time}")
                
                # Get vehicle counts
                print(f"Getting vehicle counts...")
                vehicle_counts = get_vehicle_counts(args.excel_path, args.date, start_time_str)
                
                # Compute power spectrum
                print(f"Computing power spectrum...")
                freqs, power = compute_power_spectrum(data, SAMPLING_RATE)
                
                # Plot
                print(f"Creating plot...")
                fig = plot_spectrum_with_labels(freqs, power, vehicle_counts, start_time, end_time, args.sensor_id)
                
                # Save figure
                output_filename = f"spectrum_{args.date}_{args.hour:02d}_{start_time_str.replace(':', '-')}_{args.sensor_id}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
                print(f"** Saved plot to: {output_path}")
                
                # Display plot
                plt.show()
                    
            except Exception as e:
                print(f"** Error processing {start_time_str}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete! Processed {len(args.start_time)} interval(s)")
    print(f"{'='*60}")


# Main entry point - fully parametric via command line arguments
if __name__ == "__main__":
    """
    ex: python3 spec.py -r /data/pool/c8x-98x/bridge_data -e /data/pool/c8x-98x/traffic_data/7_AID_webcam_data/Febbraio/classi.xlsx -d 20250208 -H 0 -s 030911EF_x -t 00:00:00 -t 00:15:00 -t 00:30:00 -t 00:45:00 --superimpose
    For help:
        python spectrum_analyzer.py --help
    """
    args = parse_arguments()
    main_parametric(args)