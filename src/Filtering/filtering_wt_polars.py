import argparse
import polars as pl
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def sensor_data_clip(path):
    """ Load CSV with Polars, ensuring 'time' is parsed as datetime """
    df = pl.read_parquet(path)
    return df

def filter_dc_by_mean(data, sensor_column):
    """ Remove DC component using mean """
    print(data.head())
    df = data.clone()
    mean_value = df[sensor_column].mean()
    df = df.select(
        [pl.col('time'), 
         (pl.col(sensor_column) - mean_value).alias(sensor_column)])
    return df
    

def filterby_threshold(data, threshold, sample_period, sensor_column):
    # Extract sensor data as numpy array for easier indexing
    sensor_data = data[sensor_column].to_numpy()
    filtered_indices = []
    
    i = 0
    while i < len(data):
        # Check if current value exceeds threshold
        if np.abs(sensor_data[i]) >= threshold:
            # Found start of segment
            start = i
            end = min(i + sample_period, len(sensor_data))
            
            # Extend window if values continue to exceed threshold
            while i < end:
                if np.abs(sensor_data[i]) >= threshold:
                    end = min(i + sample_period, len(sensor_data))
                i += 1
                
            # Add all indices in this segment
            filtered_indices.extend(range(start, end))
        else:
            i += 1
    
    # Create a boolean mask for filtered indices
    mask = np.zeros(len(data), dtype=bool)
    mask[filtered_indices] = True
    
    # Create a new column with filtered sensor data (zeroed where not in filtered_indices)
    filtered_values = np.zeros_like(sensor_data)
    filtered_values[filtered_indices] = sensor_data[filtered_indices]
    
    # Create the result DataFrame
    filtered_df = data.select([
        pl.col("time"),
        pl.col(sensor_column).alias("original_signal"),
        pl.lit(filtered_values).alias(sensor_column)
    ])
    filtered_df = filtered_df.to_pandas()


    plt.figure(figsize=(16, 5))
    plt.plot(data['time'], data[sensor_column], label ='original')
    plt.plot(data['time'], filtered_df[sensor_column], color='g',  label="filtered with Threshold = {}".format(threshold))
    plt.title(f"Time-series signal with the filtering threshold")
    plt.xlabel("Time")
    plt.ylabel("Acceleration:"  )
    plt.legend( )
    plt.grid()
    plt.show()

    # Calculate ratio of filtered data to original data
    signal_fil_ratio = len(filtered_indices) / len(data)
    
    return filtered_df, signal_fil_ratio


def mse_cal(data: pl.DataFrame, sensor_column: str) -> float:     
    """ Calculate Root Mean Square Error (RMSE) """
    signal_org = data["original_signal"]
    signal_fil = data[sensor_column]
    mse = (signal_org - signal_fil).pow(2).mean()
    
    rmse = np.sqrt(mse)
    print('RMSE VALUE IS:', rmse)
    return rmse

def hist(df, sensor_column):
    """ Create histogram plot """
    counts, bins, _ = plt.hist(df[sensor_column], edgecolor='black')
    
    bin_intervals = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    for i in range(len(bin_intervals)):
        print(f"Histogram Interval: {bin_intervals[i]}, Count: {counts[i]}")


def RMSE_graph(df):
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['threshold'], df['RMSE'], marker='o', linestyle='-')
    plt.xlabel('Threshold')
    plt.ylabel('RMSE')
    plt.grid()
    plt.title('Threshold vs RMSE')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process CSV files, filter sensor data using chunk and threshold")
    parser.add_argument('--file_path', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--chunk', type=int, required=True, help="Chunk size (data points in 10 ms)")
    parser.add_argument('--thresholds', type=str, required=True, help="Comma-separated list of thresholds (e.g., 0.0005,0.001,0.002)")
    args = parser.parse_args()

    path = args.file_path
    chunk = args.chunk
    thresholds = [float(x) for x in args.thresholds.split(',')]
    thresholds.sort()
    sensor_column = '0309101E_x'

    df = sensor_data_clip(path)
    df_no_dc = filter_dc_by_mean(df, sensor_column)

    print(df_no_dc.head())

    RMSE_results = []
    for threshold in thresholds:
        filtered_df = filterby_threshold(df_no_dc, threshold, chunk, sensor_column)
        rmse = mse_cal(filtered_df, sensor_column)
        RMSE_results.append(rmse)
        print(f"Threshold: {threshold}, RMSE: {rmse:.6f}")
        #hist(filtered_df, sensor_column)

    rmse_df = pl.DataFrame({'threshold': thresholds, 'RMSE': RMSE_results})
    RMSE_graph(rmse_df)
   
if __name__ == '__main__':
    main()