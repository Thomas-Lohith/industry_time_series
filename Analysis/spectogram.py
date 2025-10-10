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
    #print(df.head())
    print(sensor_column)
    signal = df[f'{sensor_column}'].dropna()
    signal = signal - signal.mean()
    df[f'{sensor_column}'] = signal
    return df

def spectogram(df, sensor_column):
    
    x = df[f'{sensor_column}']

    Fs = 100
    f, t, Sxx = signal.spectrogram(
        x, Fs, window=signal.get_window("hamming", 256), 
        nperseg=256, noverlap=64
    )
    # Convert power to decibels (dB)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)   # add epsilon to avoid log(0)

    plt.figure(figsize=(15, 8))
    pcm = plt.pcolormesh(t, f, Sxx_dB, shading="gouraud", cmap="viridis")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.ylim([0, 50])
    plt.colorbar(pcm, label='Power [dB]')

    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]
    plt.suptitle(f"Spectrogram of {sensor_column}\nStart: {start_time} | End: {end_time}")
    plt.savefig("graphs/spectrogram.png", dpi=300, bbox_inches="tight")
    plt.title(f"Spectrogram of {sensor_column}")
    plt.show()



def fft_spectrum(df, sensor_column):
    signal= df[f'{sensor_column}']
    fs = 100
    N= len(signal)
    # Compute the FFT 
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1./fs)

    # Convert to magnitude (absolute value)
    magnitude = np.abs(fft)

    plt.figure(figsize=(14, 4))
    plt.plot(freqs, magnitude, label='FFT')
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Spectrum of {sensor_column}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/fft_spectrum.png')
    plt.show()


def power_spectrum(df, sensor_column):
    #fs-frequency bins, pxx_den -power at each frequency  with the unit of v2/hz
    x = df[f'{sensor_column}']
    fs = 100

    # Welch method for PSD estimation (more averaging = smoother)
    f, Pxx = signal.welch(
        x, 
        fs=fs, 
        nperseg=256,
        scaling= 'density'       # use ~10 sec window for good averaging      # 50% overlap    # gives units of V²/Hz (PSD)
    )

    # Convert to micro units if needed (optional)
    Pxx = Pxx * 1e6

    # Focus on the most informative frequency range (0–7 Hz)
    freq_limit = 50
    mask = f <= freq_limit
    f, Pxx = f[mask], Pxx[mask]
    plt.semilogy(f, Pxx)
    plt.title(f"Power Spectrum Density - {sensor_column}")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD * 1e6 [V**2/Hz]') # 'y-PSD' should be scaled to *10e-6 
    #plt.legend()
    #plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('graphs/power_spectrum.png')
    plt.show()


### ex to run the script--->  python3 spectogram.py --path /Users/thomas/Data/Data_sensors/20250303/M001_2025-03-03_00-00-00_gg-108_int-1_th.csv --sensor 030911EF_x                               
def main():
    parser = argparse.ArgumentParser(description="create a spectogram, using sensor data using for specific columns")
    parser.add_argument('--path', type=str, required=True, help="Path to file")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name(s) to process")
    args = parser.parse_args()

    path = args.path
    sensor_column = args.sensor

    df = sensor_data(path, sensor_column)

    df = filter_dc_by_mean(df, sensor_column)

    spectogram(df[:100000], sensor_column)

    fft_spectrum(df[:100000], sensor_column)

    power_spectrum(df[:100000], sensor_column)
    
if __name__ == "__main__":
    main()