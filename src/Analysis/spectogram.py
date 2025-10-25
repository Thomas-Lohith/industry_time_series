import pandas as pd
import polars as pl
import numpy as np
from scipy import signal
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm

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
        x, Fs, window=signal.get_window("hamming", 2048), 
        nperseg=2048
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
    plt.title(f"Spectrogram of {sensor_column}")
    plt.savefig("graphs/spectrogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def spectrogram_3d_wireframe(df, sensor_column):
    """
    Generates a wireframe 3D spectrogram emphasizing structural frequency modes.
    """
    x = df[f'{sensor_column}']
    Fs = 100

    f, t, Sxx = signal.spectrogram(
        x, Fs, window=signal.get_window("hamming", 256),
        nperseg=256, noverlap=128
    )
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)

    T, F = np.meshgrid(t, f)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot as wireframe
    wire = ax.plot_wireframe(
        T, F, Sxx_dB,
        rstride=3, cstride=3,
        color='steelblue',
        linewidth=0.4
    )

    ax.set_xlabel('Time [s]', labelpad=10)
    ax.set_ylabel('Frequency [Hz]', labelpad=10)
    ax.set_zlabel('Amplitude [dB]', labelpad=10)
    ax.set_title(f'3D Spectrogram (Wireframe) - {sensor_column}', fontsize=12, pad=15)
    ax.view_init(elev=30, azim=-135)
    #plt.tight_layout()
    plt.savefig("graphs/3d_wire_spectrogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def threeD_spectogram(df, sensor_column):
    x = df[f'{sensor_column}']
    Fs = 100  # Sampling frequency

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        x, Fs, window=signal.get_window("hamming", 256),
        nperseg=256, noverlap=128  # slightly higher overlap gives smoother curves
    )

    # Convert to decibel scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)

    # Prepare meshgrid for 3D surface
    T, F = np.meshgrid(t, f)

    # Normalize for color mapping
    norm = plt.Normalize(Sxx_dB.min(), Sxx_dB.max())
    cmap = cm.get_cmap("viridis")  # same as 2D spectrogram default
    colors = cmap(norm(Sxx_dB))  

    # Create 3D figure
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot as surface
    surf = ax.plot_surface(
        T, F, Sxx_dB,
        facecolors=colors,
        linewidth=0.3,
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=False
    )

    # Label axes (consistent with physical meaning)
    ax.set_xlabel('Time (s)', labelpad=10)
    ax.set_ylabel('Frequency (Hz)', labelpad=10)
    ax.set_zlabel('Amplitude [dB]', labelpad=10)
    ax.set_title(f'3D Spectrogram - {sensor_column}', fontsize=12, pad=15)

    # Adjust 3D viewing angle (like MATLAB waterfall)
    ax.view_init(elev=30, azim=-135)

    # Add colorbar for amplitude scale
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Amplitude [dB]', rotation=270, labelpad=15)
    #plt.tight_layout()
    plt.savefig("graphs/3D_Spectrogram.png", dpi=300, bbox_inches="tight")
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
        nperseg=512,
        scaling= 'density'       # use ~10 sec window for good averaging      # 50% overlap    # gives units of V²/Hz (PSD)
    )

    # Convert to micro units if needed (optional)
    Pxx = Pxx * 1e6

    # Focus on the most informative frequency range (0–7 Hz)
    freq_limit = 20
    mask = f <= freq_limit
    f, Pxx = f[mask], Pxx[mask]

    plt.figure(figsize=(12,6))
    plt.semilogy(f, Pxx)
    plt.title(f"Power Spectrum Density - {sensor_column}")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD * 1e6 [V**2/Hz]') # 'y-PSD' should be scaled to *10e-6 
    #plt.ylim(1e-6, 1e2)
    plt.legend(fontsize=8)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('graphs/power_spectrum.png')

    plt.show()

def compare_diff_windows(df, sensor_column):
    x= df[f'{sensor_column}']
    fs = 100 #HZ

    window_types = ['hann', 'hamming', 'blackman', 'flattop', 'bartlett']
    nperseg_values = [256, 512, 1024, 2048]

    plt.figure(figsize=(12,6))

    for window in window_types:
        for nperseg in nperseg_values:
            f, pxx = signal.welch(x, fs=fs, window=window, nperseg=nperseg, scaling='density')
            plt.semilogy(f, pxx, label = f'{window}, n={nperseg}')

    plt.xlabel('Frequecy [Hz]')
    plt.ylabel('PSD [V²/Hz]')
    plt.title(f'PSD Comparison for {sensor_column}')
    plt.legend(fontsize=8)
    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('graphs/psd_window_comparison.png')
    plt.show()

def plot_comparing_windows_grid_with_peaks(df, sensor_column, save_path=None):
    """
    Compare Welch PSD for multiple window types and segment lengths,
    highlighting the top frequency peaks in each plot.
    """
    x = df[f'{sensor_column}']
    fs = 100  # Sampling frequency [Hz]
    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]

    # Window types and segment lengths to test
    window_types = ['hann', 'blackman', 'flattop']
    nperseg_values = [512, 1024, 2048, 4096, 8192] #~5.12, 10.24, 20.48, 40.96, 81.92 seconds (since fs=100 Hz).

    # Create subplot grid
    fig, axes = plt.subplots(len(window_types), len(nperseg_values), figsize=(14, 8), sharex=True, sharey=True)

    for i, window in enumerate(window_types):
        for j, nperseg in enumerate(nperseg_values):
            f, Pxx = signal.welch(
                x,
                fs=fs,
                window=window,
                nperseg=nperseg,
                scaling='density'
            )
            # Convert to micro units if needed (optional)
            Pxx = Pxx * 1e6
            # Focus on a specific frequency range (adjust as needed)
            mask = f <= 20
            f, Pxx = f[mask], Pxx[mask]

            ax = axes[i, j]
            ax.semilogy(f, Pxx, color='steelblue', linewidth=1.2)
            ax.set_title(f"{window}, n={nperseg}", fontsize=9)

            # --- Peak Detection ---
            # Find local peaks in PSD (tune height & distance if needed)
            #freq_resolution = fs / nperseg(ex. 100/1024 = 0.0976Hz)
            #distance ≈ desired_freq_spacing / freq_resolution (ex. 0.5Hz/0.0976)
            peaks, properties = signal.find_peaks(Pxx, height=np.max(Pxx)*0.02, distance=20)# np.max(Pxx)*0.05, distance=15)

            # Mark detected peaks
            ax.plot(f[peaks], Pxx[peaks], 'ro', markersize=3)
            for k in peaks[:5]:  # Annotate up to 5 most significant peaks
                ax.annotate(f"{f[k]:.1f} Hz", (f[k], Pxx[k]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, color='red', rotation=30)

            ax.grid(True, linestyle="--", alpha=0.4)

    # Shared labels
    fig.suptitle(f"PSD Comparison with Peak Detection — {sensor_column} \nStart: {start_time} | End: {end_time}", fontsize=14)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")
    for ax in axes[:, 0]:
        ax.set_ylabel('PSD * 1e6 [V**2/Hz]') # 'y-PSD' should be scaled to *10e-6

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(f'graphs/windows_comp/{sensor_column}_psd_window_comparison_with_peak_detection_20freq.png')
        plt.savefig(save_path)
        plt.show()
    else:
        plt.savefig(f'graphs/windows_comp/{sensor_column}_ psd_window_comparison_with_peak_detection_20freq.png')
        plt.show()


### ex to run the script--->  python3 spectogram.py --path /Users/thomas/Data/Data_sensors/20250303/csv_acc/M001_2025-03-03_16-00-00_gg-108_int-17_th.csv  --sensor 030911EF_x
#python3 spectogram.py --path /Users/thomas/Data/Data_sensors/20250303/csv_acc/M001_2025-03-03_16-00-00_gg-108_int-17_th.csv  --sensor 030911EF_x --date 20250303
def main():
    parser = argparse.ArgumentParser(description="create a spectogram, using sensor data using for specific columns")
    parser.add_argument('--path', type=str, required=True, help="Path to file")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name(s) to process")
    parser.add_argument('--date', type=str, required=True, help="target date")
    args = parser.parse_args()

    path = args.path
    sensor_column = args.sensor
    target_date = args.date

    df = sensor_data(path, sensor_column)

    df = filter_dc_by_mean(df, sensor_column)

    #spectogram(df[:100000], sensor_column)

    #spectrogram_3d_wireframe(df[:10000], sensor_column)

    #threeD_spectogram(df[:10000], sensor_column)

    fft_spectrum(df, sensor_column)

    power_spectrum(df, sensor_column)

    #compare_diff_windows(df[:100000], sensor_column)

    plot_comparing_windows_grid_with_peaks(df, sensor_column,
                                           save_path=f'/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Graphs/spectrum/psd_window_comparison_{target_date.replace("/", "_")}.png' )
    
if __name__ == "__main__":
    main()