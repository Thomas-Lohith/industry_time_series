import polars as pl
from datetime import datetime
import matplotlib.pyplot as plt
import time

def sensor_data_clip_polars(file_path: str):
    """
    Reads sensor data from a CSV file using Polars and returns the DataFrame and mean of a specific column.
    """
    # Load the data, parsing the 'time' column
    #date_parser = lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f')
    
    df = pl.read_csv(file_path, separator=';', has_header=True, try_parse_dates=True)
    df = df.with_columns(pl.col("time").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S:%f"))  # Parsing 'time' as datetime
    
    # Set 'time' column as index (Polars doesn't use an index but we can keep it as a column for reference)
    df = df.with_columns(pl.col("time").alias("index"))
    
    # # Compute mean of a specific sensor column (replace with actual column name)
    # mean = df.select(pl.col('0309101E_x')).mean().item()
    return df

def filter_sensor_data_by_chunk_polars(data: pl.DataFrame, sensor_column: str, chunk_size: int, threshold: float) -> pl.DataFrame:
    """
    Filters a Polars DataFrame column in chunks, retaining only values greater than or equal to a given threshold.
    """
    if sensor_column not in data.columns:
        raise ValueError(f"Column '{sensor_column}' not found in the data.")

    filtered_chunks = []
    print(data.head(10))
    # Process the data in chunks
    for start in range(0, data.height, chunk_size):
        chunk = data[start:start + chunk_size].select(sensor_column)
  
        # Filter the chunk for values greater than or equal to the threshold
        filtered_chunk = chunk.filter(chunk[sensor_column] >= threshold)
        
        if not filtered_chunk.is_empty():
            filtered_chunks.append(filtered_chunk)
    
    filtered_data = pl.concat(filtered_chunks)
    filtered_data['time'] = data['index']

    return filtered_data if filtered_chunks else pl.DataFrame()

# Input parameters
file_path = '/Users/thomas/Desktop/phd_unipv 2/Industrial_PhD/Data/20241126/csv_acc/infile.csv'
sensor_column = '0309101E_x'
chunk_size = 100
threshold = 1.0009

# Timing the Polars implementation
start = time.time()

df = sensor_data_clip_polars(file_path)
filtered_data = filter_sensor_data_by_chunk_polars(df, sensor_column, chunk_size, threshold)

end = time.time()
print(f'Runtime of Polars function: {end - start} seconds')
print(filtered_data.head(10))

# Visualization using Matplotlib
plt.figure(figsize=(16, 5))
plt.plot(df['index'].to_numpy(), df[sensor_column].to_numpy(), label='Original Data')
#plt.plot(filtered_data['index'].to_numpy(), filtered_data[sensor_column].to_numpy(), label='Filtered Data')
plt.title(f"Time-Domain Analysis of {sensor_column}")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.grid()
plt.legend()
plt.savefig('filtered_data_polars.png')
plt.show()