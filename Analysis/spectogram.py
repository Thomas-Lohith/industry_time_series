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
        df = pd.read_csv(path, sep=';')
    if path.endswith('.parquet'):
        df = pd.read_parquet(path, engine='pyarrow', columns=col)
    return df

def filter_dc_by_mean(df, sensor_column):
    print(df.head())
    print(sensor_column)
    signal = df[f'{sensor_column}'].dropna()
    signal = signal - signal.mean()
    df[f'{sensor_column}'] = signal
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
    plt.colorbar()
    plt.show()



def fft_spectrum(df, sensor_column):
    signal= df[f'{sensor_column}']
    # Compute the FFT for the first two samples
    fft_0 = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1)
    plt.figure(figsize=(14, 4))
    plt.plot(freqs, np.abs(fft_0), label='FFT Sample 0')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT of Last Two Samples')
    plt.legend()
    plt.tight_layout()
    plt.show()

### ex to run the script python3 spectogram.py 
def main():
    parser = argparse.ArgumentParser(description="create a spectogram, using sensor data using for specific columns")
    parser.add_argument('--path', type=str, required=True, help="Path to file")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name(s) to process")
    args = parser.parse_args()

    path = args.path
    sensor_column = args.sensor

    df = sensor_data(path, sensor_column)

    df = filter_dc_by_mean(df, sensor_column)

    spectogram(df[:200000], sensor_column)

    fft_spectrum(df[:200000], sensor_column)
    
if __name__ == "__main__":
    main()