import pandas as pd
#import polars as pl
import numpy as np
from scipy import signal
import argparse
import matplotlib.pyplot as plt



def compare_power_spectra(paths, sensor_column, labels=None, freq_limit=20, save_path=None):
    """
    Compare Welch Power Spectral Density (PSD) across multiple datasets.
    Each path corresponds to a different measurement (e.g., with/without traffic).
    """
    fs = 100
    nperseg = 2048
    window_type = 'hann'

    plt.figure(figsize=(12, 6))
    #colors = plt.cm.viridis(np.linspace(0, 1, len(paths)))  # colormap for multiple lines

    for i, path in enumerate(paths):
        # --- Load data ---
        if path.endswith('.csv'):
            df = pd.read_csv(path, sep=';')
        elif path.endswith('.parquet'):
            df = pd.read_parquet(path, engine='pyarrow', columns=[sensor_column, 'time'])
        else:
            print(f"Unsupported file type for {path}")
            continue

        # --- Preprocess signal ---
        x = df[sensor_column].dropna()
        x = x - x.mean()

        #--get the time stamp
        start_time = df["time"].iloc[0]
        end_time = df["time"].iloc[-1]

        # --- Compute Welch PSD ---
        f, Pxx = signal.welch(
            x,
            fs=fs,
            window='hann',
            nperseg=2048,
            scaling='density'
        )

        # Focus on low-frequency range
        mask = f <= freq_limit
        f, Pxx = f[mask], Pxx[mask]
        Pxx *= 1e6  # convert to µ-units (optional)

        # --- Plot each PSD ---
        #label = labels[i] if labels else f"Dataset {i+1}"
        label =f"Start: {start_time} | End: {end_time}"

        plt.semilogy(f, Pxx, label=label, linewidth=1.2) #color=colors[i], 

    plt.title(f"Superimposed Power Spectral Densities — {sensor_column}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD * 1e6 [V²/Hz]")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.savefig(f"graphs/{sensor_column}_superimposed_psd.png", dpi=300, bbox_inches="tight")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create PSD/spectrograms and compare multiple datasets")
    parser.add_argument('--paths', nargs='+', required=True, help="Paths to multiple files (space-separated)")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name to process")
    parser.add_argument('--labels', nargs='*', help="Optional labels for datasets")
    parser.add_argument('--date', type=str, help="Target date (optional)")
    args = parser.parse_args()

    paths = args.paths
    sensor_column = args.sensor
    labels = args.labels if args.labels else [f"Data {i+1}" for i in range(len(paths))]

    # Compare PSD across datasets
    compare_power_spectra(paths, sensor_column, labels=labels)

    
if __name__ == '__main__':
    main()
