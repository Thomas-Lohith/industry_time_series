import polars as pl
import datetime as dt
import matplotlib.pyplot as plt

def main():
    
    df = pl.read_csv('/Users/thomas/Desktop/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/infile.csv', try_parse_dates=True, has_header=True, separator= ',')
  
    print(df.head())
    print(df.shape)
    plt.figure(figsize=(16, 5))
    plt.plot(df["0309101E_x"])
    plt.title(f"Time-Domain Analysis of Sensor")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.grid()
    #plt.savefig('head_csv_ts.png')
    plt.show()
    return df


def filter_sensor_data_by_chunk(data, sensor_column, chunk_size, threshold):
 
    # Ensure the column exists in the data
    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")
    
    # Drop NaN values from the specified column
    sensor_data = data[sensor_column].drop_nulls()
    
    # Initialize an empty list to store the filtered chunks
    filtered_chunks = []
    
    # Process the data in chunks
    for start in range(0, len(sensor_data), chunk_size):
        # Define the chunk
        chunk = sensor_data[start:start + chunk_size]
        
        # Filter the chunk to keep values greater than or equal to the threshold
        filtered_chunk = chunk[chunk >= threshold]
        
        # Append the filtered chunk to the list
        filtered_chunks.append(filtered_chunk)
    
    # Concatenate all filtered chunks
    filtered_data = pl.concat(filtered_chunks)
    
    # Return the filtered data as a DataFrame
    return filtered_data.to_frame(name=sensor_column)



df = main()

#input parametrs 
sensor_column = '0309101E_x' #should be changed accordingly to the sensor preference 

chunk_size = 1000
threshold = 0.9


filtered_data = filter_sensor_data_by_chunk(df, sensor_column, chunk_size, threshold)

print(filtered_data.head())









