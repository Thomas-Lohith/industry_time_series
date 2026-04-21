import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d

# Import everything from the original analysis script
from src.analysis.vibration_analysis import (
    load_data_polars,
    parse_sensor_ids,
    filter_dc_by_mean,
    get_only_interested_duration)


def generate_frequency_anomaly(signal: np.ndarray, offset_hz: float, 
                                compression_coef: float, fs: float = 100.0,
                                keep_duration: bool = True) -> np.ndarray:
    """
    Compress frequency axis above offset_hz.
    
    Args:
        signal: 1D vibration signal
        offset_hz: Frequencies below this are unchanged
        compression_coef: Multiplier for freqs > offset_hz (0.5=compress to half, 1.0=no change)
        fs: Sampling frequency in Hz
        keep_duration: If True, return same duration with fewer samples
                      If False, return same sample count with stretched time
    
    Returns:
        Synthetic anomaly signal
    """
    fft_vals = rfft(signal)
    freqs = rfftfreq(len(signal), d=1/fs)
    
    # Compress frequency axis above offset
    compressed_freqs = freqs.copy()
    mask = freqs > offset_hz
    compressed_freqs[mask] = offset_hz + (freqs[mask] - offset_hz) * compression_coef
    
    # Maximum frequency after compression
    max_compressed_freq = compressed_freqs[-1]
    
    if keep_duration:
        # Strategy 1: Same duration, fewer samples
        new_fs = 2 * max_compressed_freq  # Nyquist
        new_n_samples = int(len(signal) * new_fs / fs)
        new_freqs = rfftfreq(new_n_samples, d=1/new_fs)
    else:
        # Strategy 2: Same sample count, stretched time
        new_n_samples = len(signal)
        new_freqs = rfftfreq(new_n_samples, d=1/fs)
    
    # Interpolate FFT to new uniform grid
    interp_real = interp1d(compressed_freqs, fft_vals.real, kind='linear', 
                           fill_value=0, bounds_error=False)
    interp_imag = interp1d(compressed_freqs, fft_vals.imag, kind='linear', 
                           fill_value=0, bounds_error=False)
    
    fft_resampled = interp_real(new_freqs) + 1j * interp_imag(new_freqs)
    
    return irfft(fft_resampled, n=new_n_samples)


def plot_anomaly_comparison(original: np.ndarray, anomaly: np.ndarray, 
                            fs: float = 100.0, offset_hz: float = 7.0,
                            anomaly_fs: float = None):
    """
    Plot original vs anomaly in time and frequency domains.
    
    Args:
        anomaly_fs: Sampling rate of anomaly signal (if different from original)
    """
    if anomaly_fs is None:
        anomaly_fs = fs
    
    # Compute FFT
    fft_orig = np.abs(rfft(original))
    fft_anom = np.abs(rfft(anomaly))
    freqs_orig = rfftfreq(len(original), d=1/fs)
    freqs_anom = rfftfreq(len(anomaly), d=1/anomaly_fs)
    
    # Time vectors
    time_orig = np.arange(len(original)) / fs
    time_anom = np.arange(len(anomaly)) / anomaly_fs
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Time domain - full signal
    axes[0, 0].plot(time_orig, original, 'b-', alpha=0.7, label='Original')
    axes[0, 0].plot(time_anom, anomaly, 'r-', alpha=0.7, label='Anomaly')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Time Domain (Full) | Orig: {len(original)} pts, Anom: {len(anomaly)} pts')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time domain - zoomed (first 5 seconds)
    zoom_samples_orig = int(5 * fs)
    zoom_samples_anom = int(5 * anomaly_fs)
    axes[0, 1].plot(time_orig[:zoom_samples_orig], original[:zoom_samples_orig], 'b-', alpha=0.7, label='Original')
    axes[0, 1].plot(time_anom[:zoom_samples_anom], anomaly[:zoom_samples_anom], 'r-', alpha=0.7, label='Anomaly')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Time Domain (First 5s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency domain - full spectrum
    axes[1, 0].semilogy(freqs_orig, fft_orig, 'b-', alpha=0.7, label='Original')
    axes[1, 0].semilogy(freqs_anom, fft_anom, 'r-', alpha=0.7, label='Anomaly')
    axes[1, 0].axvline(offset_hz, color='k', linestyle='--', alpha=0.5, label=f'Offset ({offset_hz} Hz)')
    axes[1, 0].set_xlabel('Frequency [Hz]')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Frequency Domain (Full)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency domain - zoomed (0-30 Hz)
    freq_mask_orig = freqs_orig <= 30
    freq_mask_anom = freqs_anom <= 30
    axes[1, 1].semilogy(freqs_orig[freq_mask_orig], fft_orig[freq_mask_orig], 'b-', alpha=0.7, label='Original')
    axes[1, 1].semilogy(freqs_anom[freq_mask_anom], fft_anom[freq_mask_anom], 'r-', alpha=0.7, label='Anomaly')
    axes[1, 1].axvline(offset_hz, color='k', linestyle='--', alpha=0.5, label=f'Offset ({offset_hz} Hz)')
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Frequency Domain (0-30 Hz)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser('Synthetic anomaly generator for bridge vibration data')
    parser.add_argument('--path', type = str, required=True, help= 'path for the file')
    parser.add_argument('--start_time', type=str, required=True, help= 'starting time frame interedted in')
    parser.add_argument('--duration_mins', type=float, required=True, help = 'duration in mins of time frame interested')
    parser.add_argument('--sensor',        type=str,   default=None,  help = 'Sensor ID(s) to analyze (comma-separated for multiple)')
    parser.add_argument('--offset_hz',     type=float, default=7.0)
    parser.add_argument('--compression_coef',       type=float, default=0.9)
    args = parser.parse_args()

    df, available_sensors, time_column = load_data_polars(args.path)
    sensor_columns = parse_sensor_ids(args.sensor, available_sensors)
    no_dc_df       = filter_dc_by_mean(df, sensor_columns)
    sampled_df     = get_only_interested_duration(no_dc_df, sensor_columns, time_column, args.start_time, args.duration_mins)

    for sensor in sensor_columns:
        original = sampled_df[sensor].to_numpy()

        # Strategy 1: Same duration, fewer samples
        anomaly1 = generate_frequency_anomaly(original, offset_hz=7.0, compression_coef=0.9, keep_duration=True)
        new_fs1 = len(anomaly1) / (len(original) / 100.0)
        plot_anomaly_comparison(original, anomaly1, fs=100.0, offset_hz=7.0, anomaly_fs=new_fs1)

        # Strategy 2: Same sample count, stretched time
        anomaly2 = generate_frequency_anomaly(original, offset_hz=7.0, compression_coef=0.9, keep_duration=False)
        plot_anomaly_comparison(original, anomaly2, fs=100.0, offset_hz=7.0, anomaly_fs=100.0)




        # anomaly  = generate_frequency_anomaly(original, args.offset_hz, args.compression_coef, upsample_to_original= False)
        # print(f"[{sensor}] original RMS={np.sqrt(np.mean(original**2)):.6f} | anomaly RMS={np.sqrt(np.mean(anomaly**2)):.6f}")
        # plot_anomaly_comparison(original, anomaly, offset_hz=args.offset_hz )


if __name__ == "__main__":
    main()