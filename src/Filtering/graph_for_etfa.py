import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import psutil
import os
from scipy.signal import find_peaks
import argparse
import plotly.graph_objects as go

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
        total_rows = df.select(pl.count()).item() #use .collect if we use scan_csv function
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
    
    # sensors 99,100,101
    campate2_sensor_columns = ['030911D2_x', '03091005_x', '0309101F_x']
    #sensors in order 53->52->51
    campate1b_sensor_columns = ['0309100F_x', '030910F6_x', '0309101E_x']
    #sensors in order 106->105->104
    campate1a_sensor_columns = ['030911FF_x', '030911EF_x', '03091200_x'] 
    sensor_columns = [col for col in campate1a_sensor_columns if col in df.columns]
    #sensor_columns = sensor_columns['03091200_x', '030911EF_x', '030911FF_x']
    memory_usage()
    return df, sensor_columns, time_column

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

    print("\nProcessed Data (First 5 Rows):")
    
    #print(result_df.head(1))

    return result_df

def visualize_all_sensors(df, sensor_columns, time_column, start_time, duration_mins):
    print(f"Visualizing all sensors with sample interval from {start_time} to {duration_mins} mins...")
    memory_usage()

    # Select only necessary columns and sample at specified interval
    sampled_df = (df.select([time_column] + sensor_columns)  # Keep relevant columns
    #.collect() used if you are scanning the file
    .to_pandas())
    print(f"Sampled data shape: {sampled_df.shape}")
    memory_usage()
    
    sampled_df[time_column] = pd.to_datetime(sampled_df[time_column], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    #print(sampled_df.head())

    #PLOT ONLY duration we want to analyse 
    start_time = pd.to_datetime(start_time)
    end_time = start_time + pd.Timedelta(minutes=duration_mins)
    #limit to the specific time frame
    sampled_df = sampled_df[(sampled_df[time_column]>=start_time)&(sampled_df[time_column]<=end_time)]

    # Create figure
    plt.figure(figsize=(16, 9))
    plt.plot(sampled_df[time_column], sampled_df[sensor_columns[0]], linewidth=1, alpha=0.7)
    # Format the plot
    plt.title('Acceleration Data from Multiple Sensors')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    sensor = sensor_columns[0]
    return sampled_df, sensor
    
def filterby_threshold(data, threshold, sample_period, sensor_column):

    #print(data.head(10))
    # Extract sensor data
    sensor_data = data[sensor_column]
  
    filtered_indices = []
    filtered_df = pd.DataFrame(columns=['time', 'original_signal', sensor_column])
    
    i = 0
    while i < len(data):
        # Check if current value exceeds threshold
        if np.abs(sensor_data.iloc[i]) >= threshold:
            # Found start of segment
            start = i
            end = min(i + sample_period, len(sensor_data))
            
            # Extend window if values continue to exceed threshold
            while i < end:
                if np.abs(sensor_data.iloc[i]) >= threshold:
                    end = min(i + sample_period, len(sensor_data))
                i += 1
                
            # Add all indices in this segment
            filtered_indices.extend(range(start, end))
        else:
            i += 1
    
    filtered_sensor_data = sensor_data.iloc[filtered_indices]
    filtered_df['time'] = data['time']
    filtered_df[sensor_column] = filtered_sensor_data
    filtered_df[sensor_column] = filtered_df[sensor_column].fillna(0)
    filtered_df['original_signal'] = sensor_data

    plt.figure(figsize=(16, 9))
    plt.plot(data['time'], data[sensor_column], label ='original')
    plt.plot(data['time'], filtered_df[sensor_column],label='filtered', color ='y', linewidth=1, alpha=0.7)
    # Format the plot
    plt.axhline(threshold, label='threshold', color='red', linestyle='--')
    plt.axhline(-threshold, color='red', linestyle='--')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Acceleration', fontsize=18)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ETFA.png')
    plt.show()


    # Calculate ratio of filtered data to original data
    signal_fil_ratio = len(filtered_indices) / len(data[sensor_column])
    return filtered_df, signal_fil_ratio

def main():
    parser = argparse.ArgumentParser('Analyse the vibration realting the dyanmic weighing data')
    parser.add_argument('--path', type = str, required=True, help= 'path for the file')
    parser.add_argument('--start_time', type=str, required=True, help= 'starting time frame interedted in')
    parser.add_argument('--duration_mins', type=float, required=True, help = 'duration in mins of time frame interested')
    parser.add_argument('--chunk', type=int, required=True, help="chunk size (number of data points in window)")
    parser.add_argument('--threshold', type=float, required=True, help="Comma-separated list of thresholds (e.g., 0.0005,0.001,0.002)")
 
    #ex for script vibration_analysis.py --path path_to_folder --start_time  
    args = parser.parse_args()
    path = args.path # Path to your parquet file
    start_time = args.start_time 
    duration_mins = args.duration_mins
    chunk = args.chunk
    threshold = args.threshold
    
      
    # Load data using Polars
    df, sensor_columns, time_column = load_data_polars(path)

    # Process the filtered data
    no_dc_df = filter_dc_by_mean(df, sensor_columns)

    # visualise each sensor in campate for a sample interval
    sample_df, sensor = visualize_all_sensors(no_dc_df, sensor_columns, time_column, start_time, duration_mins)
    
    filterby_threshold(sample_df, threshold, chunk, sensor)
  
    print("Analysis complete!")
    memory_usage()

if __name__ == "__main__":
    main()