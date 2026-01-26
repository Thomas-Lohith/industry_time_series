import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime, timedelta
import glob
import os

def find_csv_file(root_folder, date_str, hour):
    """
    Find the CSV file for the specified date and hour.
    Format: YYYYMMDD/csv_acc/M001_YYYY-MM-DD_HH-00-00_gg-83_int-*_th.csv
    
    Args:
        root_folder: Root directory path
        date_str: Date in YYYYMMDD format
        hour: Hour (0-23)
    
    Returns:
        Path to CSV file
    """
    hour1=int(hour+1)
    # Convert date format from YYYYMMDD to YYYY-MM-DD
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    formatted_date = f"{year}-{month}-{day}"
    
    # Build the search pattern
    csv_folder = os.path.join(root_folder, date_str, "csv_acc")
    pattern = os.path.join(csv_folder, f"M001_{formatted_date}_{hour:02d}-00-00_gg-83_int-{hour1:02d}_th.csv")
    
    # Find matching files
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No CSV file found matching pattern: {pattern}")
    
    return matching_files[0]


def extract_15min_data(csv_path, sensor_id, start_time_str):
    """
    Extract 15 minutes of data from the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        sensor_id: Column name for the sensor (e.g., '03091002_x')
        start_time_str: Starting time in format 'HH:MM:SS'
    
    Returns:
        Extracted data array and actual time range
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, sep=';')
    #print(df.head())
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
    """
    Get vehicle counts for the specified 15-minute interval.
    
    Args:
        excel_path: Path to Excel file with vehicle counts
        date_str: Date in YYYYMMDD format (e.g., '20250201')
        start_time_str: Starting time in format 'HH:MM:SS'
    
    Returns:
        Dictionary with vehicle counts by type
    """
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
    print(vehicle_counts)
    return vehicle_counts


def compute_power_spectrum(data, sampling_rate):
    """
    Compute the power spectrum of the signal.
    
    Args:
        data: Signal data array
        sampling_rate: Sampling rate in Hz
    
    Returns:
        frequencies, power spectrum
    """
    # Remove mean
    data = data - np.mean(data)

    freqs, power = signal.welch(
            data,
            fs=sampling_rate,
            window='hann',
            nperseg=2048,
            scaling='density'
        )
    
    freq_limit = 20
    mask = freqs <= freq_limit
    freqs, power = freqs[mask], power[mask]
    power *= 1e6  # convert to µ-units (optional)
    
    return freqs, power


def plot_spectrum_with_labels(freqs, power, vehicle_counts, start_time, end_time, sensor_id):
    """
    Plot the power spectrum with vehicle count labels.
    
    Args:
        freqs: Frequency array
        power: Power spectrum array
        vehicle_counts: Dictionary of vehicle counts
        start_time: Start datetime
        end_time: End datetime
        sensor_id: Sensor identifier
    """
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


def main(root_folder, excel_path, date_str, hour, sensor_id, start_time_str):
    """
    Main function to process data and create spectrum plot.
    
    Args:
        root_folder: Root directory containing date folders
        excel_path: Path to Excel file with vehicle counts
        date_str: Date in YYYYMMDD format
        hour: Hour (0-23)
        sensor_id: Sensor column name
        start_time_str: Start time in 'HH:MM:SS' format
    """
    # Find and read CSV file
    print(f"Searching for CSV file...")
    csv_path = find_csv_file(root_folder, date_str, hour)
    print(f"Found: {csv_path}")
    
    # Extract 15-minute data
    print(f"Extracting 15-minute data for sensor {sensor_id}...")
    data, start_time, end_time = extract_15min_data(csv_path, sensor_id, start_time_str)
    print(f"Extracted {len(data)} samples from {start_time} to {end_time}")
    
    # Determine sampling rate (assuming 10-second intervals = 0.1 Hz, but calculate from data)
    # From the image, it looks like 10-second intervals, so 6 samples per minute = 0.1 Hz
    sampling_rate = 100  # Adjust based on actual data
    
    # Get vehicle counts
    print(f"Getting vehicle counts...")
    vehicle_counts = get_vehicle_counts(excel_path, date_str, start_time_str)
    print(f"Vehicle counts: {vehicle_counts}")
    
    # Compute power spectrum
    print(f"Computing power spectrum...")
    freqs, power = compute_power_spectrum(data, sampling_rate)
    
    # Plot
    print(f"Creating plot...")
    fig = plot_spectrum_with_labels(freqs, power, vehicle_counts, start_time, end_time, sensor_id)
    
    # Save figure
    output_filename = f"spectrum_{date_str}_{hour:02d}_{start_time_str.replace(':', '-')}_{sensor_id}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_filename}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # User inputs
    ROOT_FOLDER = "/Users/thomas/Data/Data_sensors"  # Update this path
    EXCEL_PATH = "/Users/thomas/Data/7_AID_webcam_data/Febbraio/classi.xlsx"  # Update this path
    DATE = "20250208"  # YYYYMMDD format
    HOUR = 23  # 0-23
    SENSOR_ID = "030911EF_x"  # Sensor column name
    START_TIME = "23:00:00"  # HH:MM:SS format
    
    # Run the analysis
    main(ROOT_FOLDER, EXCEL_PATH, DATE, HOUR, SENSOR_ID, START_TIME)