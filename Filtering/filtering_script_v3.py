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

def filter_dc_by_mean(data):
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
    plt.figure(figsize=(16, 5))
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
    plt.plot(data['time'], data[sensor_column], color='g', label ='original')
    plt.plot(data['time'], filtered_df[sensor_column], color='r', label ='filtered')
    plt.title(f"Time-Domain Analysis of Sensor")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.grid()
    plt.show()
    return filtered_df

def mse_cal(data):
    signal_org = data['original_signal']
    signal_fil = data[sensor_column] 
    MSE = np.square(np.subtract(signal_org,signal_fil)).mean() 
    RMSE = math.sqrt(MSE)
    return RMSE


sensor_column = '0309101E_x'
path = '/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/M001_2024-11-26_18-00-00_gg-9_int-19_th.csv'
# sampling_rate = 50  # Hz (10 ms sampling interval)
# cutoff_freq = 0.1 # Very low frequency for DC component

# #parametrs for threshold script
chunk = 20 #should be converted into 2000 datapoints as 20sec
threshold = 0.0025 
#thresholds = [0.0005, 0.001,0.018, 0.002, 0.005]

df = sensor_data_clip(path)
print(df[sensor_column].head(5))
print(df.shape)

df_no_dc = filter_dc_by_mean(df)
print(df_no_dc[sensor_column].head(5))
print(df_no_dc.shape)

filtered_df = filterby_threshold(df_no_dc, threshold, chunk, sensor_column)
print(filtered_df[sensor_column][1430:1450])
print(filtered_df.shape)

RMSE = mse_cal(filtered_df)
print('RMSE VALUE IS:', RMSE)

np.set_printoptions(precision = 6, suppress = True)
counts, bins,_ = plt.hist(df[sensor_column], edgecolor ='black')  # Specify the number of bins
plt.show()

print("counts:",np.float64(counts))
print("bins:", bins)
# Intervals between bins
bin_intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
print("Histogram Intervals:",np.float64(bin_intervals))