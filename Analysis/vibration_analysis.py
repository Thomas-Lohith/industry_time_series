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
    
    print(result_df.head(1))

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
    #print(sampled_df.head())

    #PLOT ONLY duration we want to analyse 
    start_time = pd.to_datetime(start_time)
    print('cehck:', start_time)
    end_time = start_time + pd.Timedelta(minutes=duration_mins)

    #limit to the specific time frame
    sampled_df = sampled_df[(sampled_df[time_column]>=start_time)&(sampled_df[time_column]<=end_time)]
    print(sampled_df.head())

    #if plot == 0:
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
    plt.tight_layout()
    #plt.savefig('capmate_1a_sensor_vibrations.svg', format= 'svg')
    plt.show()
    
    fig = go.Figure()

    # Loop over each sensor and add a trace to the figure
    for sensor in sensor_columns:
        fig.add_trace(go.Scatter(
            x=sampled_df[time_column],  # Time data for the x-axis
            y=sampled_df[sensor],  # Sensor data for the y-axis
            mode='lines',  # Line plot
            name=sensor,  # Label the trace with the sensor name
            line=dict(width=1),  # Line width
            opacity=0.7  # Line transparency
        ))

    # Update layout for the plot
    fig.update_layout(
        title='Acceleration Data from Sensors in campate_1a',  # Title of the plot
        xaxis_title='Time',  # Label for the X-axis
        yaxis_title='Acceleration',  # Label for the Y-axis
        legend_title='Sensors',  # Title for the legend
        legend=dict(
            x=0.5,  # Position of the legend
            y=-0.15,  # Below the plot
            xanchor='center',  # Center the legend horizontally
            yanchor='top',  # Align the top of the legend
            orientation='h',  # Horizontal orientation
            traceorder='normal',  # Order in which the traces are displayed
            font=dict(size=12)  # Font size for legend
        ),
        #template='plotly_dark',  # Optional: dark template for styling
        showlegend=True,  # Show the legend
        hovermode='closest',  # Show closest data point on hover
        margin=dict(b=60),  # Adjust the bottom margin to fit legend
    )

    # Display the plot
    fig.show()
    fig.write_html(f'/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Graphs/events_with_less_trafic/vibration_data_{start_time}.html')
    print("All sensors visualization saved to all_sensors_acceleration.png")
    memory_usage()


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
    #plt.savefig('sensor_histograms.png', dpi=300)
    plt.show()
    plt.close()
    
    print("Sensor histograms saved to sensor_histograms.png")
    memory_usage()

def count_signal_peaks(df: pl.DataFrame, sensor_column: str, height_threshold: float = None, distance: int = None) -> tuple[int, np.ndarray]:

    # Convert the sensor data to numpy array
    signal = df.select(pl.col(sensor_column)).collect().to_numpy().flatten()
    
    # If height threshold is not provided, use 2 standard deviations
    if height_threshold is None:
        height_threshold = 2 * np.std(signal)
    
    # If distance is not provided, use 1% of the signal length
    if distance is None:
        distance = len(signal) // 5000
    
    # Find peaks
    peaks, _ = find_peaks(signal, height=height_threshold, distance=distance)
    
    return len(peaks), peaks

def visualize_peaks(df: pl.DataFrame, sensor_column: str, time_column: str, sample_interval: int = 36000):
        # Get peak information
    num_peaks, peak_indices = count_signal_peaks(df, sensor_column)
    print(f"Found {num_peaks} peaks in {sensor_column}")
    
    # Sample the data and covert it to pandas df for visualization 
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
    visualize_all_sensors(no_dc_df, sensor_columns, time_column, start_time, duration_mins)
    
    # Count and visualize peaks for each sensor
    # for sensor in sensor_columns:
    #     num_peaks, _ = count_signal_peaks(no_dc_df, sensor)
    #     print(f"Number of peaks in {sensor}: {num_peaks}")
    #     visualize_peaks(no_dc_df, sensor, time_column, sample_interval=6000)
    
    print("Analysis complete!")
    memory_usage()

if __name__ == "__main__":
    main()




    #instructions to run this parametric scripts:
    #check wether the parameters correctly matching the format(for ex: the date and month should be interchanegd from the format of weighing data)  
    # python3 vibration_analysis.py --path /Users/thomas/Data/ --start_time '2025/03/07 01:05:00' --duration_mins 25 
