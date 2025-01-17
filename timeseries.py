import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import time
from scipy.fft import fft, ifft, fftfreq
def sensor_data_clip():
    date_parser= lambda x:datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f')
    df = pd.read_csv('/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/infile.csv', sep=';', parse_dates= ['time'], date_parser=date_parser, index_col='time')
    # plt.figure(figsize=(16, 5))
    # plt.plot(df.index, df['0309101E_x'])
    # plt.title(f"Time-Domain Analysis of Sensor")
    # plt.xlabel("Time")
    # plt.ylabel("Acceleration")
    # plt.grid()
    # plt.savefig('head_csv_ts.png')
    # plt.show()
    return df

def filter_sensor_data_by_chunk(data, sensor_column, chunk_size, threshold):
 
    # Ensure the column exists in the data
    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")
    
    sensor_data = data[sensor_column].dropna()
    filtered_chunks = []
    # Process the data in chunks
    for start in range(0, len(sensor_data), chunk_size):
        # Define the chunk
        chunk = sensor_data.iloc[start:start + chunk_size]
        # Filter the chunk to keep values greater than or equal to the threshold
        filtered_chunk = chunk[chunk >= threshold] 
        # Append th e filtered chunk to the list
        filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks
    filtered_data = pd.concat(filtered_chunks)
    print(filtered_data.info())
    return filtered_data.to_frame(name=sensor_column)

def filterby_threshold(data, threshold, duration):

    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")
    
    sensor_data = data[sensor_column].dropna()
    filtered_chunks = []

   

def frequency_conversion(df, sensor_column):
    # Preprocess the data (detrending)
    timedomain_signal = df[sensor_column].dropna()
    timedomain_signal = timedomain_signal - timedomain_signal.mean()
    fourier = np.abs(np.fft.fft(timedomain_signal.values))
    n = timedomain_signal.size
    timestep = 0.005
    freq = np.fft.fftfreq(n, timestep)
    freq = freq[:n // 2]
    fourier = fourier[:n // 2]
    return fourier, freq


#start = time.time()
df = sensor_data_clip()
#input parametrs 
sensor_column = '0309101E_x' #should be changed accordingly to the sensor preference 
chunk_size = 20
threshold = 1.003

filtered_data = filter_sensor_data_by_chunk(df, sensor_column, chunk_size, threshold)
#end = time.time()
#print ('runtime of pandas fun:', end - start)
print(filtered_data.head())
print(filtered_data.tail())

#Time graph of the signal
plt.figure(figsize=(16, 5))
plt.plot(filtered_data.index, filtered_data[sensor_column], label = 'filtered_data', color='r', linestyle='--')
plt.plot(df.index, df[sensor_column], label= 'original')
plt.title(f"Time-Domain Analysis of Sensor")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.grid()
plt.savefig(f'time_domain {sensor_column}.png')
plt.show()

fourier, freq = frequency_conversion(df, sensor_column)
#Frequency of the signal
plt.figure(figsize=(10,6))
plt.plot(freq, fourier)
plt.title('frequency spectrum')
plt.xlabel('freq(Hz)')
plt.ylabel('Amplitude')
plt.show()