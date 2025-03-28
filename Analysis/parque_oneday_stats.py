import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import psutil
import os
from scipy.signal import find_peaks

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
    df = pl.scan_parquet(filepath)    
    # Count total rows without loading everything to memory
    total_rows = df.select(pl.len()).collect().item()
    print(f"Total rows: {total_rows:,}")
    
    # Get column names for sensor data (assuming pattern ends with _z for vertical direction)
    sensor_columns = [col for col in df.collect_schema().keys() if col != 'time']
    
    # Try to identify time column
    time_column_candidates = ['time', 'timestamp', 'date']
    time_column = next((col for col in df.collect_schema().keys() if col.lower() in time_column_candidates), None)
    
    if not time_column:
        # If no obvious time column name found, assume the first column that's not a sensor column
        all_columns = list(df.collect_schema().keys())
        remaining_columns = [col for col in all_columns if col not in sensor_columns]
        if remaining_columns:
            time_column = remaining_columns[0]
        else:
            raise ValueError("Could not identify a time column. Please specify the time column name manually.")
    
    #print(f"Found {len(sensor_columns)} sensor columns: {sensor_columns}")
    print(f"Using '{time_column}' as the time column")
    
    # sensors 99,100,101
    campate2_sensor_columns = ['030911D2_x', '03091005_x', '0309101F_x']
    campate1_sensor_columns = ['03091200_x', '030911EF_x', '030911FF_x'] 
    sensor_columns = [col for col in campate2_sensor_columns if col in df.columns]
    #sensor_columns = sensor_columns['03091200_x', '030911EF_x', '030911FF_x']
    memory_usage()
    return df, sensor_columns, time_column

def filter_dc_by_mean(df: pl.DataFrame, sensor_columns: list[str]) -> pl.DataFrame:
    """
    Remove DC offset from specified sensor columns by subtracting the mean of each column.
    
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame.
    sensor_columns : list[str]
        List of column names containing sensor data to process.
    
    Returns:
    --------
    pl.DataFrame
        DataFrame with DC offset removed from specified sensor columns.
    """
    # Clone the DataFrame to avoid modifying the original
    result_df = df.clone()
    
    # Process only specified sensor columns, ignoring 'time'
    for col in sensor_columns:
            # Compute the mean as a scalar
            col_mean = df.select(pl.col(col).mean()).collect().item()
            
            # Subtract mean and update the column
            result_df = result_df.with_columns(
                (pl.col(col) - col_mean).alias(col)
            )

    print("\nProcessed Data (First 5 Rows):")
    
    #print(result_df.head(1))

    return result_df

def visualize_all_sensors(df, sensor_columns, time_column, sample_interval):
    """
    Create a plot showing all sensors' z-axis acceleration data
    Using sampling to avoid memory issues and plotting delays
    """
    print(f"Visualizing all sensors with sample interval of {sample_interval}...")
    memory_usage()


    # Select only necessary columns and sample at specified interval
    sampled_df = (df.select([time_column] + sensor_columns)  # Keep relevant columns
    .slice(0, sample_interval)  # Select the first 2000 rows
    .collect()
    .to_pandas()
    )
    

    print(f"Sampled data shape: {sampled_df.shape}")
    memory_usage()
    
    # Convert timestamp to datetime if it's not already
    #if pd.api.types.is_string_dtype(sampled_df[time_column]):
    #print(sampled_df['time'][350990:360010])
    sampled_df[time_column] = pd.to_datetime(sampled_df[time_column], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    
    print(sampled_df.head())

    # Create figure
    plt.figure(figsize=(16, 9))
    

    # Plot each sensor
    for sensor in sensor_columns:
        plt.plot(sampled_df[time_column], sampled_df[sensor], label=sensor, linewidth=1, alpha=0.7)
    
    # Format the plot
    plt.title('Acceleration Data from Multiple Sensors')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis labels
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S:%f'))
    # plt.xticks(rotation=45)
    #plt.gca().set_xticklabels([label.get_text()[:-3] for label in plt.gca().get_xticklabels()])

    plt.tight_layout()
    plt.savefig('all_sensors_acceleration.png', dpi=300)
    plt.show()
    plt.close()
    
    print("All sensors visualization saved to all_sensors_acceleration.png")
    memory_usage()

def compare_sensors_statistics(df, sensor_columns, time_column):
    """Calculate and compare basic statistics for each sensor"""
    print("Calculating statistics for each sensor...")
    memory_usage()
    
    # Calculate statistics for each sensor
    # stats = df.select([
    #     pl.col(col).mean().alias(f"{col}_mean"),
    #     pl.col(col).std().alias(f"{col}_std"),
    #     pl.col(col).min().alias(f"{col}_min"),
    #     pl.col(col).max().alias(f"{col}_max"),
    #     (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_range")
    # ] for col in sensor_columns).collect()
    
    stats_expressions = []
    for col in sensor_columns:
        stats_expressions.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
            (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_range")
        ])

    # Select using unpacking
    stats = df.select(*stats_expressions).collect()

    # Convert to a more readable format
    stats_dict = {
        "Sensor": [],
        "Mean": [],
        "Std Dev": [],
        "Min": [],
        "Max": [],
        "Range": []
    }
    
    for col in sensor_columns:
        stats_dict["Sensor"].append(col)
        stats_dict["Mean"].append(stats[f"{col}_mean"][0])    # Extracting first row value
        stats_dict["Std Dev"].append(stats[f"{col}_std"][0])
        stats_dict["Min"].append(stats[f"{col}_min"][0])
        stats_dict["Max"].append(stats[f"{col}_max"][0])
        stats_dict["Range"].append(stats[f"{col}_range"][0])
    
    stats_df = pd.DataFrame(stats_dict)
    print("\nSensor Statistics:")
    print(stats_df.head())
    memory_usage()
    
    # Visualize sensor statistics
    plt.figure(figsize=(14, 10))
    
    # Plot mean values
    plt.subplot(2, 2, 1)
    plt.bar(stats_df["Sensor"], stats_df["Mean"])
    plt.title("Mean Acceleration by Sensor")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=90)
    
    # Plot standard deviation
    plt.subplot(2, 2, 2)
    plt.bar(stats_df["Sensor"], stats_df["Std Dev"])
    plt.title("Standard Deviation by Sensor")
    plt.ylabel("Std Dev")
    plt.xticks(rotation=90)
    
    # Plot ranges
    plt.subplot(2, 2, 3)
    plt.bar(stats_df["Sensor"], stats_df["Range"])
    plt.title("Range (Max-Min) by Sensor")
    plt.ylabel("Range")
    plt.xticks(rotation=90)
    
    # Plot min and max together
    plt.subplot(2, 2, 4)
    x = np.arange(len(stats_df["Sensor"]))
    width = 0.35
    plt.bar(x - width/2, stats_df["Min"], width, label="Min")
    plt.bar(x + width/2, stats_df["Max"], width, label="Max")
    plt.title("Min and Max Acceleration by Sensor")
    plt.ylabel("Value")
    plt.xticks(x, stats_df["Sensor"], rotation=90)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sensor_statistics.png', dpi=300)
    #plt.show()
    plt.close()
    
    print("Sensor statistics visualization saved to sensor_statistics.png")
    memory_usage()
    
    return stats_df

def visualize_sensor_histograms(df, sensor_columns, bins=50):
    """Create histograms for each sensor to analyze distribution"""
    print("Creating histograms for each sensor...")
    memory_usage()
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(sensor_columns))))
    
    # Create figure
    plt.figure(figsize=(16, 16))
    
    # Sample data for histogram using filter instead of sample
    # For lazy frames, we need to use a different approach than sample()
    n = 100  # Take every nth row
    sampled_df = df.select(sensor_columns).filter(pl.arange(0, pl.len()).mod(n) == 0).collect()
    
    # Plot histogram for each sensor
    for i, sensor in enumerate(sensor_columns, 1):
        plt.subplot(grid_size, grid_size, i)
        plt.hist(sampled_df[sensor], bins=bins, alpha=0.7)
        plt.title(sensor)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensor_histograms.png', dpi=300)
    plt.close()
    
    print("Sensor histograms saved to sensor_histograms.png")
    memory_usage()

def count_signal_peaks(df: pl.DataFrame, sensor_column: str, height_threshold: float = None, distance: int = None) -> tuple[int, np.ndarray]:
    """
    Count the number of peaks in a sensor's vibration signal.
    
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame containing the sensor data
    sensor_column : str
        Name of the column containing the sensor data
    height_threshold : float, optional
        Minimum height of peaks to be counted. If None, automatically determined
    distance : int, optional
        Minimum number of samples between peaks. If None, automatically determined
    
    Returns:
    --------
    tuple[int, np.ndarray]
        Number of peaks and array of peak indices
    """
    # Convert the sensor data to numpy array
    signal = df.select(pl.col(sensor_column)).collect().to_numpy().flatten()
    
    # If height threshold is not provided, use 2 standard deviations
    if height_threshold is None:
        height_threshold = 2 * np.std(signal)
    
    # If distance is not provided, use 1% of the signal length
    if distance is None:
        distance = len(signal) // 200
    
    # Find peaks
    peaks, _ = find_peaks(signal, height=height_threshold, distance=distance)
    
    return len(peaks), peaks

def visualize_peaks(df: pl.DataFrame, sensor_column: str, time_column: str, sample_interval: int = 36000):
    """
    Visualize the peaks found in the sensor signal.
    
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame
    sensor_column : str
        Name of the column containing the sensor data
    time_column : str
        Name of the column containing the time data
    sample_interval : int
        Number of samples to plot
    """
    # Get peak information
    num_peaks, peak_indices = count_signal_peaks(df, sensor_column)
    print(f"Found {num_peaks} peaks in {sensor_column}")
    
    # Sample the data for visualization
    sampled_df = (df.select([time_column, sensor_column])
                 .slice(0, sample_interval)
                 .collect()
                 .to_pandas())
    
    # Convert timestamp to datetime
    sampled_df[time_column] = pd.to_datetime(sampled_df[time_column], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    
    # Create the plot
    plt.figure(figsize=(16, 9))
    plt.plot(sampled_df[time_column], sampled_df[sensor_column], label='Signal', alpha=0.7)
    
    # Plot peaks
    peak_times = sampled_df[time_column].iloc[peak_indices[peak_indices < sample_interval]]
    peak_values = sampled_df[sensor_column].iloc[peak_indices[peak_indices < sample_interval]]
    plt.scatter(peak_times, peak_values, color='red', label='Peaks', zorder=5)
    
    plt.title(f'Signal Peaks for {sensor_column}')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    #plt.savefig(f'peaks_{sensor_column}.png', dpi=300)
    #plt.close()
    
    print(f"Peak visualization saved to peaks_{sensor_column}.png")

def main():
    # Path to your parquet file
    parquet_file = "/Users/thomas/Data/20241126/csv_acc/sample_17hr_26_11_2024.parquet"  # Update with your actual file path
    
    # Load data using Polars
    df, sensor_columns, time_column = load_data_polars(parquet_file)

    no_dc_df = filter_dc_by_mean(df, sensor_columns)
    
    # Visualize all sensors (sampled for performance)
    visualize_all_sensors(no_dc_df, sensor_columns, time_column, sample_interval=360000)
    
    # Count and visualize peaks for each sensor
    for sensor in sensor_columns:
        num_peaks, _ = count_signal_peaks(no_dc_df, sensor)
        print(f"Number of peaks in {sensor}: {num_peaks}")
        visualize_peaks(no_dc_df, sensor, time_column, sample_interval=360000)
    
    # Calculate and compare sensor statistics
    #stats_df = compare_sensors_statistics(no_dc_df, sensor_columns, time_column)
    
    # Create histograms for each sensor
    visualize_sensor_histograms(no_dc_df, sensor_columns)
    
    print("Analysis complete!")
    memory_usage()

if __name__ == "__main__":
    main()