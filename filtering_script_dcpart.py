import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq

def sensor_data_clip():
    date_parser= lambda x:datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f')
    df = pd.read_csv('/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/infile.csv', sep=';', parse_dates= ['time'], date_parser=date_parser, index_col=None)
    return df

def filterby_threshold(data, threshold, sample_period, sensor_column):

    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")
     
    sensor_data = data[sensor_column].dropna()
    time_indices = data['time']
    filtered_indices = []
    filtered_df = pd.DataFrame(columns= ['filtered_time', 'filtered_sensor_signal'])
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
    filtered_df['filtered_sensor_signal'] = filtered_sensor_data
    return filtered_df

def frequency_conversion(df, sensor_column):
    # Preprocess the data (detrending)
    timedomain_signal = df[sensor_column].dropna()
    timedomain_signal = timedomain_signal - timedomain_signal.mean()
    fourier = fft (timedomain_signal.values)
    fourier = np.abs(fourier)
    #fourier = np.abs(np.fft.fft(timedomain_signal.values))
    n = timedomain_signal.size
    timestep = 0.005
    freq = np.fft.fftfreq(n, timestep)
    freq = freq[:n // 2]
    fourier = fourier[:n // 2]
    dc_removed_fft = ifft(fourier)
    return fourier, freq, dc_removed_fft

def high_pass_filter(data, cutoff_freq, sampling_rate):
    
    signal = data[sensor_column].dropna()
    time_indices = data['time']
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normalized_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)



def graphs(sensor_column, df, dc_removed_fft):
    fourier, freq, dc_removed_fft = frequency_conversion(df, sensor_column)
    #Frequency of the signal
    plt.figure(figsize=(10,6))
    plt.plot(freq, fourier)
    plt.plot(freq, dc_removed_fft, color = 'r')
    plt.title('frequency spectrum')
    plt.xlabel('freq(Hz)')
    plt.ylabel('Amplitude')
    plt.show()



if __name__ == '__main__':
    
    df = sensor_data_clip()
    print(df.head())
    #input parametrs 
    sensor_column = '0309101E_x' #should be changed accordingly to the sensor preference 
    chunk_size = 20
    sample_period =20# should be cpnverted into 2000 datapoints as 20sec
    threshold = 1.005
    interval_time= 10
    sampling_rate = 50  # Hz (10 ms sampling interval)
    cutoff_freq = 10  # Very low frequency for DC component
    dc_removed_high_pass = high_pass_filter(df, cutoff_freq, sampling_rate)
    plt.plot(dc_removed_high_pass)
    plt.show()
    filtered_data = filterby_threshold(df, threshold, sample_period, sensor_column)

    fourier, freq, dc_removed_fft = frequency_conversion(df, sensor_column)