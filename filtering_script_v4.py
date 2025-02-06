import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math
import time

def sensor_data_clip(path):
    date_parser= lambda x:datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f')
    date_parser_2= lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    df = pd.read_csv(path, sep=';', parse_dates= ['time'], date_parser=date_parser, index_col=None)
    return df

def filter_dc_by_mean(data, sensor_column):
    signal = data[sensor_column].dropna()
    signal = signal - signal.mean()
    data[sensor_column] = signal
    return data

def filterby_threshold(data, threshold, sample_period, sensor_column):
    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")
    sensor_data = data[sensor_column].dropna()
    #time_indices = data['time']
    filtered_indices = []
    filtered_df = pd.DataFrame(columns= ['time','original_signal', sensor_column])
    i=0
    while i < len(data):
        if np.abs(sensor_data.iloc[i])>= threshold:
            threshold_flag = True
            start = i
            end = min(i+sample_period, len(sensor_data)) 
            while i < end:
                if np.abs(sensor_data.iloc[i])>= threshold:
                    end = min(i+sample_period, len(sensor_data)) 
                i +=1
            filtered_indices.extend(range(start,end))
        else:
            i+=1
    #filtered_time_index = time_indices[filtered_indices]  
    # #to get only filtered values including time indices and corresponding sensor readings
    filtered_sensor_data = sensor_data.iloc[filtered_indices]
    filtered_df['time'] = data['time']
    filtered_df[sensor_column] = filtered_sensor_data
    filtered_df[sensor_column] = filtered_df[sensor_column].fillna(0)
    filtered_df['original_signal'] = sensor_data

    plt.figure(figsize=(16, 5))
    plt.plot(data['time'], data[sensor_column], label ='original')
    plt.plot(data['time'], filtered_df[sensor_column], color='g',  label="filtered with Threshold = {}".format(threshold    ))
    plt.title(f"Time-series signal with the filtering threshold")
    plt.xlabel("Time")
    plt.ylabel("Acceleration:")
    plt.legend()
    plt.grid()
    plt.show()
    signal_fil_ratio = len(filtered_indices)/len(data[sensor_column])
    return filtered_df,signal_fil_ratio

def mse_cal(data, sensor_column):
    signal_org = data['original_signal']
    signal_fil = data[sensor_column] 
    np.set_printoptions(precision = 9, suppress = True)
    MSE = np.square(np.subtract(signal_org,signal_fil)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE

def hist(df,sensor_column):
    np.set_printoptions(precision = 6, suppress = True)
    counts, bins,_ = plt.hist(df[sensor_column] )  # Specify the number of bins
    bin_intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    for i in range(len(bins)-1):
        print("Histogram Intervals:", np.float64(bin_intervals[i]),  "count", counts[i])

def main():
    parser = argparse.ArgumentParser(description="Process CSV files, flter sensor data uisng chunk and threshold ")
    parser.add_argument('--file_path', type=str, required = True, help="List of CSV files to process")
    parser.add_argument('--chunk', type=int, required = True, help="chunk size(number of data points in 10 ms)")
    parser.add_argument('--thresholds', type=str, required = True, help="Comma-separated list of thresholds (e.g., 0.0005,0.001,0.002)")
    #parser.add_argument("--sensors", type=str, required=True, help="Sensor column name to process.")
    args = parser.parse_args()
    path = args.file_path
    chunk = args.chunk
    thresholds = [float(x) for x in args.thresholds.split(',')]
    thresholds.sort()
    sensor_column = '0309101E_x'

    df = sensor_data_clip(path)
    df_no_dc = filter_dc_by_mean(df, sensor_column)
  
    RMSE_results = []
    signal_filtering_ratio = []

    for threshold in thresholds:
        filtered_df, signal_fil_ratio = filterby_threshold(df_no_dc, threshold, chunk, sensor_column)
        rmse = mse_cal(filtered_df, sensor_column)
        RMSE_results.append(rmse)
        signal_filtering_ratio.append(signal_fil_ratio)
        print(f"Threshold: {threshold}, RMSE: {rmse:.6f}, signal_filtering_ratio: {signal_fil_ratio:.6f}")
        hist(filtered_df, sensor_column)
        #filtered_df.to_csv("filtered_data.csv", index=False)

    rmse_df = pd.DataFrame({'threshold': thresholds, 'RMSE': RMSE_results, 'signal_ratio': signal_filtering_ratio}, columns= ['threshold', 'RMSE', 'signal_ratio'])

    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    ax1.plot(rmse_df['threshold'], rmse_df['RMSE'], marker = 'o', linestyle = '-')
    ax2.plot(rmse_df['threshold'], rmse_df['signal_ratio'])
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('signal_filtering_ratio')
    plt.grid()
    plt.title('threshold vs rmse vs signal_ratio')
    plt.show()


if __name__ == '__main__':
    main()

    
