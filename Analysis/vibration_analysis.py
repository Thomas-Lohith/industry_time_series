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
    campate1a_sensor_columns = ['030911FF_x', '030911EF_x', '03091200_x', '03091155_z', '03091207_x', '03091119_z'] 
    #sensor column on whole brideg
    all_campate_sensor_columns = ['030911FF_x', '03091017_z', '03091113_x', '0309123B_z', '03091111_z', '03091003_x'] 
    sensor_columns = [col for col in all_campate_sensor_columns if col in df.columns]
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

    # print("\nProcessed Data (First 5 Rows):")
    # print(result_df.head())

    return result_df

def visualize_all_sensors(df, sensor_columns, time_column, start_time, duration_mins):
    print(f"Visualizing sensors in one campate with sample interval from {start_time} to {duration_mins} mins...")
    memory_usage()

    # Select only necessary columns and sample at specified interval
    sampled_df = (df.select([time_column] + sensor_columns)  # Keep relevant columns
    #.collect() used if you are scanning the file
    .to_pandas()
    )
    print(f"Sampled data shape: {sampled_df.shape}")
    memory_usage()
    
    sampled_df[time_column] = pd.to_datetime(sampled_df[time_column], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    #print('chek the date format:', sampled_df.head(1))

    #PLOT ONLY duration we want to analyse 
    start_time = pd.to_datetime(start_time)
    #print('check:', start_time)
    end_time = start_time + pd.Timedelta(minutes=duration_mins)

    #limit to the specific time frame
    sampled_df = sampled_df[(sampled_df[time_column]>=start_time)&(sampled_df[time_column]<=end_time)]
    #print('intereseted time dataframe-start', sampled_df.head(1))
    #print('limited time frame dataframe-end', sampled_df.tail(1))


 
    # Choose the sensors you want to plot with matplolib
    sensor_list = sensor_columns[:6]  # or list(vertical_columns.values())[:6]
    n = len(sensor_list)
    cols = 3
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
    plt.savefig('graphs/subplots.png')
    plt.show()

    # Save to file
    fig.savefig(f'/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Graphs/events_with_less_trafic/vibration_data_{start_time}.png', dpi=300)
        
#     fig = go.Figure()
#    # Choose the sensors you want to plot
#     sensor_list = sensor_columns[:6] # Adjust the number as needed

#     # Create subplot structure: 2 rows, 3 columns for 6 sensors
#     n = len(sensor_list)
#     cols = 3
#     rows = -(-n // cols)

#     fig = make_subplots(
#         rows=rows,
#         cols=cols,
#         subplot_titles=sensor_list  # Use sensor names as titles
#     )
#     # Loop and add each sensor's trace to the appropriate subplot
#     for i, sensor in enumerate(sensor_list):
#         row = (i // cols) + 1
#         col = (i % cols) + 1
        
#         fig.add_trace(
#             go.Scatter(
#                 x=sampled_df[time_column],
#                 y=sampled_df[sensor],
#                 mode='lines',
#                 name=sensor,
#                 line=dict(width=1),
#                 opacity=0.8
#             ),
#             row=row,
#             col=col
#         )

#     # Update layout
#     fig.update_layout(
#         title_text="Sensor Vibration Plots (Vertical Direction)",
#         showlegend=False
#     )
#     # Display the plot
#     fig.show()
#     fig.write_html(f'/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Graphs/events_with_less_trafic/vibration_data_{start_time}.html')
#     print("All sensors visualization saved to all_sensors_acceleration.png")
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
    plt.savefig('graphs/multisensor_spectogram.png')
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

def main():
    parser = argparse.ArgumentParser('Analyse the vibration realting the dyanmic weighing data')
    parser.add_argument('--path', type = str, required=True, help= 'path for the file')
    parser.add_argument('--start_time', type=str, required=True, help= 'starting time frame interedted in')
    parser.add_argument('--duration_mins', type=float, required=True, help = 'duration in mins of time frame interested')

    #ex for script RUNNING: python3 vibration_analysis.py --path  --start_time --duration_mins
    args = parser.parse_args()
    path = args.path # Path to your parquet file
    start_time = args.start_time 
    duration_mins = args.duration_mins
      
    # Load data using Polars
    df, sensor_columns, time_column = load_data_polars(path)

    # Process the filtered data
    no_dc_df = filter_dc_by_mean(df, sensor_columns)

    # visualise each sensor in campate for a sample interval
    sampled_df = visualize_all_sensors(no_dc_df, sensor_columns, time_column, start_time, duration_mins)
    
    multi_sensor_spectrogram(sampled_df, sensor_columns, cols=3)

    #visualize_sensor_histograms(sampled_df, sensor_columns, bins=50)
   
    print("Analysis complete!")
    memory_usage()

if __name__ == "__main__":
    main()
    #instructions to run this parametric scripts:
    #check wether the parameters correctly matching the format(for ex: the date and month should be interchanegd from the format of weighing data)  
    # python3 vibration_analysis.py --path /Users/thomas/Data/Data_sensors/20250307/csv_acc/M001_2025-03-07_01-00-00_gg-112_int-2_th.csv --start_time '2025/03/07 01:05:00' --duration_mins 5