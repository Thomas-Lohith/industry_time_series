import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq


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

def filterby_threshold(data, threshold, sample_period, sensor_column):
    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")
     
    sensor_data = data[sensor_column].dropna()
    time_indices = data['time']
    filtered_indices = []
    filtered_df = pd.DataFrame(columns= ['filtered_time', sensor_column])
    i=0
    plt.figure(figsize=(16, 5))
    plt.plot(data['time'], data[sensor_column], label= 'original', alpha=0.7)
    
    threshold_flag = False
    while i < len(data):
        if sensor_data.iloc[i]>= threshold :
            threshold_flag = True
            start = i
            end = min(i+sample_period, len(sensor_data)) 
            while i < end:
                if sensor_data.iloc[i]>= threshold :
                    end = min(i+sample_period, len(sensor_data)) 
                i +=1
            filtered_indices.extend(range(start,end))
        else:
            threshold_flag = False
            i+=1
    filtered_time_index = time_indices[filtered_indices]
    filtered_sensor_data = sensor_data.iloc[filtered_indices]
    plt.scatter(filtered_time_index, filtered_sensor_data, color='red', s=10, label='Filtered Signal')
    plt.title(f"Time-Domain Analysis of Sensor")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.grid()
    plt.show()
    filtered_df['filtered_time'] = filtered_time_index
    filtered_df[sensor_column] = filtered_sensor_data
    return filtered_df


sensor_column = '0309101E_x'
path = '/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/M001_2024-11-26_18-00-00_gg-9_int-19_th.csv'
sampling_rate = 50  # Hz (10 ms sampling interval)
cutoff_freq = 0.5 # Very low frequency for DC component
#parametrs for threshold script
sample_period =20 #should be cpnverted into 2000 datapoints as 20sec
threshold = 0.001


df = sensor_data_clip(path)
print(df[sensor_column].head(15))
print(df.shape)

df_no_dc = high_pass_filter(df, cutoff_freq, sampling_rate)
print(df_no_dc[sensor_column].head(15))
print(df_no_dc.shape)

filtered_df = filterby_threshold(df_no_dc, threshold, sample_period, sensor_column)
print(filtered_df[sensor_column].head(25))
print(filtered_df.shape)