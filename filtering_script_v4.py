import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
import math

def sensor_data_clip(path):
    date_parser= lambda x:datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f')
    df = pd.read_csv(path, sep=';', parse_dates= ['time'], date_parser=date_parser, index_col=None)
    return df

def high_pass_filter(data, cutoff_freq, sampling_rate):
    signal = data[sensor_column].dropna()
    time_indices = data['time']
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normalized_cutoff, btype='high', analog=False)
    dc_removed_signal = filtfilt(b, a, signal)
    data[sensor_column] = dc_removed_signal
    return data

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
    filtered_sensor_data = []
    filtered_indices = []
    filtered_df = pd.DataFrame(columns= ['time','original_signal', sensor_column])
    i=0
    #plt.figure(figsize=(16, 5))
    threshold_flag = False
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
            threshold_flag = False
            i+=1

    #filtered_time_index = time_indices[filtered_indices]
    filtered_sensor_data = sensor_data.iloc[filtered_indices]
    filtered_df['time'] = data['time']
    filtered_df[sensor_column] = filtered_sensor_data
    filtered_df[sensor_column] = filtered_df[sensor_column].fillna(0)
    filtered_df['original_signal'] = sensor_data
    # plt.plot(data['time'], data[sensor_column], color='g', label ='original')
    # plt.plot(data['time'], filtered_df[sensor_column], color='r', label ='filtered')
    # plt.title(f"Time-Domain Analysis of Sensor")
    # plt.xlabel("Time")
    # plt.ylabel("Acceleration")
    # plt.grid()
    # plt.show()
    return filtered_df


def mse_cal(data, sensor_column):
    signal_org = data['original_signal']
    signal_fil = data[sensor_column] 
    np.set_printoptions(precision = 9, suppress = True)
    #mean = np.mean(signal_fil) 
    MSE = np.square(np.subtract(signal_org,signal_fil)).mean() 
    RMSE = math.sqrt(MSE)
    print('RMSE VALUE IS:', RMSE)
    #print('mean VALUE IS:', mean)
    return RMSE

def hist(df,sensor_column):
    np.set_printoptions(precision = 6, suppress = True)
    counts, bins,_ = plt.hist(df[sensor_column], edgecolor ='black')  # Specify the number of bins
    plt.show()

    # print("counts:",np.float64(counts))
    # print("bins:", bins)
    # Intervals between bins
    bin_intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        print("Histogram Intervals:", np.float64(bin_intervals[i]),  "count", counts[i])

def main():
    parser = argparse.ArgumentParser(description="Process CSV files, flter sensor data uisng chunk and threshold ")
    parser.add_argument('--file_path', type=str, required = True, help="List of CSV files to process")
    parser.add_argument('--chunk', type=int, required = True, help="chunk size(number of data points in 10 ms)")
    parser.add_argument('--thresholds', type=str, required = True, help="Comma-separated list of thresholds (e.g., 0.0005,0.001,0.002)")
    args = parser.parse_args()

    path = args.file_path
    chunk = args.chunk
    thresholds = [float(x) for x in args.thresholds.split(',')]
    thresholds.sort()
    sensor_column = '0309101E_x'

    # csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]
    # if not csv_files:
    #     print("No CSV files found in the provided folder.")
    #     exit()

    # Loop through each file and process it
    # for file_name in csv_files:
    #     path = os.path.join(path, file_name)
    #     print(f"\nProcessing file: {file_name}")
    df = sensor_data_clip(path)
    df_no_dc = filter_dc_by_mean(df, sensor_column)
  
    RMSE_results = []
    for threshold in thresholds:
        filtered_df = filterby_threshold(df_no_dc, threshold, chunk, sensor_column)
        print(filtered_df.shape)
        rmse = mse_cal(filtered_df, sensor_column)
        RMSE_results.append(rmse)
        print(f"Threshold: {threshold}, RMSE: {rmse:.6f}")
        hist(filtered_df, sensor_column)
        #filtered_df.to_csv("filtered_data.csv", index=False)

    rmse_df = pd.DataFrame({'threshold': thresholds, 'RMSE': RMSE_results}, columns= ['threshold', 'RMSE'])

    plt.figure(figsize=(10,5))
    plt.plot(rmse_df['threshold'], rmse_df['RMSE'], marker = 'o', linestyle = '-')
    plt.xlabel('Thresholsd')
    plt.ylabel('rmse')
    plt.grid()
    plt.title('threshold vs rmse')
    plt.show()


# sensor_column = '0309101E_x'
# #path = '/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/M001_2024-11-26_18-00-00_gg-9_int-19_th.csv'
# # sampling_rate = 50  # Hz (10 ms sampling interval)
# # cutoff_freq = 0.1 # Very low frequency for DC component
# # #parametrs for threshold script
# chunk = 20 #should be converted into 2000 datapoints as 20sec
# threshold = 0.0025 
# #thresholds = [0.0005, 0.001,0.018, 0.002, 0.005]
main()

    
