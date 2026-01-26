import pandas as pd
#import polars as pl
import numpy as np
from scipy import signal
import argparse
import matplotlib.pyplot as plt
import os 



def load_and_process_data(file_path, sheet_name=0):
    """Load Excel data and process it"""
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    
    # Get all the columns 
    #df = df.iloc[:, :2]
    df.iloc[:, :1] = df['DateTime']
    
    # Extract date and time
    df['Date'] = df['DateTime'].apply(extract_date_from_datetime)
    df['Time'] = df['DateTime'].apply(extract_time_from_datetime)
    
    # Remove rows with missing data
    df = df.dropna(subset=['Date', 'Time'])
    return df


def compare_power_spectra(paths, sensor_column,freq_limit=20, save_path=None):
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
    args = parser.parse_args()

    paths = args.paths
    sensor_column = args.sensor

    # Compare PSD across datasets
    compare_power_spectra(paths, sensor_column)


    # Interactive mode
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print(f"{'='*60}")
    user_input = input("\nWould you like to analyze a particular date? (y/n): ").strip().lower()
    
    if user_input == 'y':
        user_date = input("Enter date (DD/MM/YYYY): ").strip()
        user_window_hour = input("Enter time window in hour (e.g., 0, 1, 2...23): ").strip()

        try:
            time_window = int(user_window_hour)

        except ValueError:
            print("Invalid time window. Using default 15 minutes.")

    
if __name__ == '__main__':
    main()


#write a program to plot spectrum graph of 15 mins samples to match the sampling rate of the telecamera data(vehicle count and type)
#each graph should show the spectrum in 15min interval with the vehicle count as label below the graph
#scripts: it should take an input of the {date}--to get the folder with this date name in root folder, {number}-to match the hour of that date in that date folder(csv_acc).
#ex:Format: YYYYMMDD/csv_acc/M001_YYYY-MM-DD_HH-00-00_gg-*_int-*_th.csv
#--->search the {date} in the {input_folder_path1(root_folder)}. [root folder has files with date as name and each date folder has sub folder csv_acc and this folder has 24 files one for each hour ]
#--->take the 15 min interval(with input of {sensor_id}) from this selected hour file
# in order to take the exact 15 min interval we take the input {starting time} and calculate the 15 min from that time_str
#--->for the label:search for the date in the date column in the {input_folder_path2(path to excel sheet)} and take the values each [column(vehical_count)] and [column_name(vehicle_type)] for the same 15 min interval
# Now create a graph of power spectrum or the givem 15 min sample with label of each vehicle type and vehicle count 

#example for power specctra caluclation