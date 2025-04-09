import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt, mpld3
import time

def sensor_data_clip(path, sensor_column):
    # Load data from parquet file with specified columns
    col = [sensor_column, 'time']
    df = pd.read_parquet(path, engine='pyarrow', columns=col)
    return df

def filter_dc_by_mean(data, sensor_column):
    # Remove DC bias by subtracting the mean
    signal = data[sensor_column]
    signal = signal - signal.mean()
    data[sensor_column] = signal
    return data

def filterby_threshold(data, threshold, sample_period, sensor_column):
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
    
    # Optional: Visualize each filtered result
    plt.figure(figsize=(16, 5))
    plt.plot(data['time'], data[sensor_column], label ='original')
    plt.plot(data['time'], filtered_df[sensor_column], color='g', label="filtered with Threshold = {}".format(threshold))
    plt.title(f"Time-series signal with the filtering threshold")
    plt.hlines(y=threshold, label='threshold = {}'.format(threshold), colors='r', linestyles='--', xmin=0, xmax=len(data))
    plt.xlabel("Time")
    plt.ylabel("Acceleration:")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate ratio of filtered data to original data
    signal_fil_ratio = len(filtered_indices) / len(data[sensor_column])
    
    return filtered_df, signal_fil_ratio

def rmse_cal(data, sensor_column):
    # Renamed to rmse_cal would be better, but keeping original name for compatibility
    signal_org = data['original_signal']
    signal_fil = data[sensor_column]
    np.set_printoptions(precision=9, suppress=True)
    MSE = np.square(np.subtract(signal_org, signal_fil)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

def printhist(df, sensor_column):
    sensor_data = df[sensor_column].to_numpy()
    counts, bins = np.histogram(sensor_data, bins=50)
    bin_intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    # Generate histogram interval strings
    s = [f"Histogram Intervals: {np.float64(bin_intervals[i])},  count: {int(counts[i])} \n" for i in range(len(counts))]
    # Print each interval
    for line in s:
        print(line.strip())
    # Write to file
    with open('histogram.txt', 'w') as h:
        h.writelines(s)
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sensor_data, bins=50, edgecolor='black')
    plt.title(f'Histogram of {sensor_column}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('histogram.png')

def RMSE_graph(rmse_df, sensor_column):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    ax1.plot(rmse_df['threshold'], rmse_df['RMSE'], marker='o', linestyle='-')
    ax2.plot(rmse_df['threshold'], rmse_df['signal_ratio'], marker='o', color='r')
    ax1.set_xlabel(f'Threshold: {sensor_column}')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('signal_filtering_ratio')
    plt.grid()
    plt.title('threshold vs rmse vs signal_ratio')
    plt.savefig('rmse_graph_pandas.png')
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Process parquet files, filter sensor data using chunk and threshold")
    parser.add_argument('--file_path', type=str, required=True, help="Path to parquet file to process")
    parser.add_argument('--chunk', type=int, required=True, help="chunk size (number of data points in window)")
    parser.add_argument('--thresholds', type=str, required=True, help="Comma-separated list of thresholds (e.g., 0.0005,0.001,0.002)")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name(s) to process")
    args = parser.parse_args()
    
    path = args.file_path
    chunk = args.chunk
    thresholds = [float(x) for x in args.thresholds.split(',')]
    thresholds.sort()
    sensor_column = args.sensor

    # Load and preprocess data
    df = sensor_data_clip(path, sensor_column)
    df_no_dc = filter_dc_by_mean(df[:10000], sensor_column)
  
    RMSE_results = []
    signal_filtering_ratio = []

    # Process each threshold
    for threshold in thresholds:
        filtered_df, signal_fil_ratio = filterby_threshold(df_no_dc, threshold, chunk, sensor_column)
        rmse = rmse_cal(filtered_df, sensor_column)
        RMSE_results.append(rmse)
        signal_filtering_ratio.append(signal_fil_ratio)
        print(f"Threshold: {threshold}, RMSE: {rmse:.6f}, signal_filtering_ratio: {signal_fil_ratio:.6f}")
        
    # Generate histogram of original data
    printhist(df_no_dc, sensor_column)

    # Create summary dataframe and plot results
    rmse_df = pd.DataFrame({
        'threshold': thresholds, 
        'RMSE': RMSE_results, 
        'signal_ratio': signal_filtering_ratio
    }, columns=['threshold', 'RMSE', 'signal_ratio'])
    
    RMSE_graph(rmse_df, sensor_column)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")