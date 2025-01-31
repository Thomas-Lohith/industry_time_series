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

def filter_dc_by_mean(data):
    signal = data[sensor_column].dropna()
    signal = signal - signal.mean()
    data[sensor_column] = signal
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
# sampling_rate = 50  # Hz (10 ms sampling interval)
# cutoff_freq = 0.1 # Very low frequency for DC component
# #parametrs for threshold script

# Parameters to explore
thresholds = [0.0005, 0.001, 0.002, 0.005, ]
chunk_sizes = [200, 1000, 2000, 5000]

# Store results
results = []

df = sensor_data_clip(path)

df_no_dc = filter_dc_by_mean(df)

for threshold in thresholds:
    for chunk_size in chunk_sizes:
        filtered_df = filterby_threshold(df_no_dc, threshold, chunk_size, sensor_column)
        # Calculate space savings
        original_points = len(df_no_dc[sensor_column].dropna())
        filtered_points = len(filtered_df[sensor_column].dropna())
        space_savings = 100 * (1 - filtered_points / original_points)
        
        # Store result
        results.append({
            'Threshold': threshold, 
            'Chunk Size': chunk_size, 
            'Original Points': original_points, 
            'Filtered Points': filtered_points, 
            'Space Savings (%)': space_savings,
        })
        

# Display results
results_df = pd.DataFrame(results)
print(results_df.head())

# Save to CSV for documentation
#results_df.to_csv('filtering_signal_preservation_results.csv', index=False)