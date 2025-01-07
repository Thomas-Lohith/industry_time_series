import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import time

def sensor_data_clip():
    date_parser= lambda x:datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f')

    df = pd.read_csv('/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/M001_2024-11-26_18-00-00_gg-9_int-19_th.csv', sep=';', parse_dates= ['time'], date_parser=date_parser, index_col='time')
    
    # plt.figure(figsize=(16, 5))
    # plt.plot(df.index, df['0309101E_x'])
    # plt.title(f"Time-Domain Analysis of Sensor")
    # plt.xlabel("Time")
    # plt.ylabel("Acceleration")
    # plt.grid()
    # plt.savefig('head_csv_ts.png')
    # plt.show()
    mean = df['0309101E_x'].mean()

    return df, mean



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
    
    # Return the filtered data as a DataFrame
    return filtered_data.to_frame(name=sensor_column)


start = time.time()
df, mean = sensor_data_clip()


#input parametrs 
sensor_column = '0309101E_x' #should be changed accordingly to the sensor preference 

chunk_size = 2000
threshold = 1.01


filtered_data = filter_sensor_data_by_chunk(df, sensor_column, chunk_size, threshold)

end = time.time()
print ('runtime of pandas fun:', end - start)
print(filtered_data.head())
print(filtered_data.tail())

plt.figure(figsize=(16, 5))
plt.plot(filtered_data.index, filtered_data['0309101E_x'])
plt.title(f"Time-Domain Analysis of Sensor")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.grid()
plt.savefig('head_csv_ts.png')
plt.show()