import pandas as pd
import polars as pl
import numpy as np
from scipy import signal
import argparse
import matplotlib.pyplot as plt

def sensor_data(path, sensor_column):
    # Load data from parquet file with specified columns
    col = [sensor_column, 'time']
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    if path.endswith('.parquet'):
        df = pd.read_parquet(path, engine='pyarrow', columns=col)
    return df

def filter_dc_by_mean(df, sensor_column):
    signal = df[sensor_column].dropna()
    signal = signal - signal.mean()
    df[sensor_column] = signal
    return df

def spectogram(df, sensor_column):
    
    x = df[f'{sensor_column}']
    
    Fs = 100
    f, t, Sxx = signal.spectrogram(x, Fs, window=signal.get_window('hamming', 256), nperseg=256, noverlap=128)
    plt.figure(figsize=(15,8))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.ylim([0, 50])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="create a spectogram, using sensor data using for specific columns")
    parser.add_argument('--path', type=str, required=True, help="Path to file")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name(s) to process")
    args = parser.parse_args()

    path = args.path
    sensor_column = args.sensor

    df = sensor_data(path, sensor_column)

    df = filter_dc_by_mean(df, sensor_column)

    spectogram(df[:2000], sensor_column)
    
if __name__ == "__main__":
    main()