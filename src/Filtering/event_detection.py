"""
Event-Based Unsupervised Detection for Bridge Monitoring

Uses vertical acceleration data from a CSV file.
Pipeline: load → preprocess → event detection → feature extraction → K-Means clustering.
Each data point represents one vehicle crossing event.
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"

FEATURE_NAMES = [
    "duration_sec",
    "start_time_min",
    "mean",
    "std",
    "var",
    "min",
    "max",
    "range",
    "rms",
    "skew",
    "kurtosis",
    "peak_pos",
    "peak_neg",
    "peak_overall",
    "energy",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "envelope_mean",
    "envelope_max",
    "envelope_std",
    "zcr",
    "p25",
    "p50",
    "p75",
    "iqr",
    "dom_freq",
    "dom_freq_mag",
    "mean_freq",
    "spectral_centroid",
    "spectral_std",
    "spectral_rolloff",
    "psd",
    "spectral_entropy",
    "energy_low",
    "energy_mid",
    "energy_high",
]


def date_hour_folder_from_filename(input_path: Path) -> str:
    """Extract date and hour from CSV filenames like M001_2025-03-18_00-00-00_gg-6_int-1_th.csv."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-\d{2}-\d{2}", input_path.stem)
    if match:
        date_str, hour_str = match.groups()
        return f"{date_str.replace('-', '')}_{hour_str}"
    return input_path.stem


def load_data(input_path: Path, sensor_column: str, delimiter: str) -> pd.DataFrame:
    """Load and validate acceleration data from a single CSV file."""
    print(f"Loading CSV: {input_path}")

    data = pd.read_csv(input_path, sep=delimiter)
    if sensor_column not in data.columns:
        raise ValueError(f"CSV must contain '{sensor_column}' column!")

    data = data[[sensor_column]].copy()
    data = data.dropna(subset=[sensor_column])
    data[sensor_column] = pd.to_numeric(data[sensor_column], errors="coerce")
    data = data.dropna(subset=[sensor_column])

    if len(data) == 0:
        raise ValueError(f"No valid data found in column '{sensor_column}'!")

    data["acceleration_z"] = data[sensor_column] - data[sensor_column].mean()
    print(f"Valid data points: {len(data)}")
    return data


def preprocess_signal(
    acceleration_z: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply band-pass filtering."""
    acceleration = acceleration_z.reshape(-1, 1).astype(np.float64)
    signal_raw = acceleration.copy()

    b, a = butter(4, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="band")
    acceleration = np.ascontiguousarray(acceleration, dtype=np.float64)

    if np.any(np.isnan(acceleration)) or np.any(np.isinf(acceleration)):
        print("Warning: NaN or Inf values found. Replacing with 0.")
        acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)

    signal_filtered = filtfilt(b, a, acceleration, axis=0)

    return signal_filtered, signal_raw


def plot_preprocessing(
    signal_raw: np.ndarray,
    signal_filtered: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    output_path: Path,
) -> None:
    """Save preprocessing pipeline visualization."""
    n_samples_plot = min(90000, len(signal_raw))
    time_axis = np.arange(n_samples_plot)

    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    fig.suptitle(
        "Signal Preprocessing Pipeline - Vertical Acceleration (Z-direction)",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].plot(time_axis, signal_raw[:n_samples_plot, 0], linewidth=0.8, color="#1f77b4", alpha=0.7)
    axes[0, 0].set_title("Step 1: Raw Signal", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Acceleration (m/s²)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(signal_raw[:, 0], bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Step 1: Raw Signal Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Acceleration (m/s²)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 0].plot(time_axis, signal_filtered[:n_samples_plot, 0], linewidth=0.8, color="#ff7f0e", alpha=0.7)
    axes[1, 0].set_title("Step 2: After Butterworth Band-Pass Filter", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Sample Index")
    axes[1, 0].set_ylabel("Acceleration (m/s²)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(signal_filtered[:, 0], bins=50, color="#ff7f0e", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("Step 2: Filtered Signal Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Acceleration (m/s²)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[2, 0].plot(time_axis, signal_raw[:n_samples_plot, 0], linewidth=1, alpha=0.5, label="Raw", color="#1f77b4")
    axes[2, 0].plot(
        time_axis, signal_filtered[:n_samples_plot, 0], linewidth=1.5, alpha=0.8, label="Filtered", color="#ff7f0e"
    )
    axes[2, 0].set_title("Comparison: Raw vs Filtered", fontsize=12, fontweight="bold")
    axes[2, 0].set_xlabel("Sample Index")
    axes[2, 0].set_ylabel("Acceleration (m/s²)")
    axes[2, 0].legend(loc="upper right")
    axes[2, 0].grid(True, alpha=0.3)

    f_raw, psd_raw = welch(signal_raw[:, 0], fs=fs, nperseg=2048, window="hann")
    f_filt, psd_filt = welch(signal_filtered[:, 0], fs=fs, nperseg=2048, window="hann")
    axes[2, 1].semilogy(f_raw, psd_raw, linewidth=1.5, alpha=0.7, label="Raw", color="#1f77b4")
    axes[2, 1].semilogy(f_filt, psd_filt, linewidth=2, alpha=0.9, label="Filtered", color="#ff7f0e")
    axes[2, 1].axvline(x=low_hz, color="r", linestyle="--", alpha=0.7, linewidth=2, label=f"Low cutoff ({low_hz} Hz)")
    axes[2, 1].axvline(x=high_hz, color="r", linestyle="--", alpha=0.7, linewidth=2, label=f"High cutoff ({high_hz} Hz)")
    axes[2, 1].set_title("Frequency Domain: Power Spectral Density", fontsize=12, fontweight="bold")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_ylabel("PSD (dB/Hz)")
    axes[2, 1].legend(loc="upper right", fontsize=8)
    axes[2, 1].grid(True, alpha=0.3, which="both")
    axes[2, 1].set_xlim([0, 25])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Preprocessing visualization saved: {output_path}")


def detect_events(
    signal: np.ndarray,
    fs: float,
    threshold_percentile: float = 85,
    smooth_sec: float = 0.5,
    min_event_sec: float = 1.0,
    merge_gap_sec: float = 1.5,
) -> tuple[list[tuple[int, int]], float, np.ndarray]:
    """Detect vehicle passage events from the signal envelope."""
    envelope = np.abs(signal)
    smooth_n = max(1, int(fs * smooth_sec))
    envelope_smooth = uniform_filter1d(envelope, size=smooth_n)

    threshold = np.percentile(envelope_smooth, threshold_percentile)
    is_active = envelope_smooth > threshold

    min_event_n = int(min_event_sec * fs)
    merge_gap_n = int(merge_gap_sec * fs)

    events: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(is_active):
        if active and start is None:
            start = i
        elif not active and start is not None:
            events.append((start, i))
            start = None
    if start is not None:
        events.append((start, len(is_active)))

    merged: list[list[int]] = []
    for s, e in events:
        if not merged:
            merged.append([s, e])
        elif s - merged[-1][1] <= merge_gap_n:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    event_segments = [(s, e) for s, e in merged if (e - s) >= min_event_n]
    if not event_segments:
        raise ValueError(
            "No events detected with current parameters. "
            "Try lowering threshold_percentile or min_event_sec."
        )

    print(f"Detected {len(event_segments)} events (threshold p{threshold_percentile}: {threshold:.4f})")
    return event_segments, threshold, envelope_smooth


def plot_event_detection(
    signal: np.ndarray,
    event_segments: list[tuple[int, int]],
    envelope_smooth: np.ndarray,
    threshold: float,
    threshold_percentile: float,
    fs: float,
    output_path: Path,
) -> None:
    """Save event detection visualization."""
    durations = [(e - s) / fs for s, e in event_segments]
    time_samples = np.arange(len(signal)) / fs / 60

    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    fig.suptitle("Event Detection - Vehicle Passage Identification", fontsize=16, fontweight="bold")

    axes[0].plot(time_samples, signal, linewidth=0.5, alpha=0.6, label="Filtered Signal")
    for i, (start, end) in enumerate(event_segments):
        axes[0].axvspan(
            start / fs / 60,
            end / fs / 60,
            alpha=0.3,
            color="red",
            label="Detected Event" if i == 0 else "",
        )
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Filtered Signal with Detected Events", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_samples, envelope_smooth, linewidth=1, label="Smoothed Envelope", color="blue")
    axes[1].axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold (p{threshold_percentile})",
    )
    for start, end in event_segments:
        axes[1].axvspan(start / fs / 60, end / fs / 60, alpha=0.2, color="green")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Envelope Amplitude")
    axes[1].set_title("Envelope and Detection Threshold", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    event_ids = np.arange(len(event_segments))
    axes[2].bar(event_ids, durations, color="skyblue", edgecolor="black")
    axes[2].axhline(
        y=np.mean(durations),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(durations):.1f}s",
    )
    axes[2].set_xlabel("Event ID")
    axes[2].set_ylabel("Duration (seconds)")
    axes[2].set_title(f"Event Durations (Total: {len(event_segments)} events)", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout() 
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Event detection visualization saved: {output_path}")


def extract_event_features(
    event_segments: list[tuple[int, int]],
    acceleration: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, list[dict]]:
    """Extract time- and frequency-domain features for each detected event."""
    features: list[np.ndarray] = []
    event_info: list[dict] = []

    for event_id, (start, end) in enumerate(event_segments):
        seg = acceleration[start:end, 0]
        duration_sec = (end - start) / fs
        start_time_min = start / fs / 60

        mean_val = np.mean(seg)
        std_val = np.std(seg)
        var_val = np.var(seg)
        min_val = np.min(seg)
        max_val = np.max(seg)
        range_val = max_val - min_val
        rms_val = np.sqrt(np.mean(seg**2))
        skew_val = skew(seg)
        kurt_val = kurtosis(seg)

        peak_pos = np.max(seg)
        peak_neg = np.abs(np.min(seg))
        peak_overall = np.max(np.abs(seg))
        energy = np.sum(seg**2)

        crest_factor = peak_overall / (rms_val + 1e-10)
        shape_factor = rms_val / (np.mean(np.abs(seg)) + 1e-10)
        impulse_factor = peak_overall / (np.mean(np.abs(seg)) + 1e-10)

        envelope_mean = np.mean(np.abs(seg))
        envelope_max = np.max(np.abs(seg))
        envelope_std = np.std(np.abs(seg))

        zero_crossings = np.sum(np.diff(np.sign(seg)) != 0)
        zcr = zero_crossings / len(seg)

        p25 = np.percentile(seg, 25)
        p50 = np.percentile(seg, 50)
        p75 = np.percentile(seg, 75)
        iqr = p75 - p25

        spec = np.abs(fft(seg))
        spec = spec[: len(spec) // 2]
        freqs = np.linspace(0, fs / 2, len(spec))

        dom_freq_idx = np.argmax(spec)
        dom_freq = freqs[dom_freq_idx]
        dom_freq_magnitude = spec[dom_freq_idx]

        mean_freq = np.mean(spec)
        spectral_centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-10)
        spectral_std = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * spec) / (np.sum(spec) + 1e-10)
        )

        cumsum_energy = np.cumsum(spec**2)
        rolloff_threshold = 0.85 * cumsum_energy[-1]
        spectral_rolloff_idx = np.where(cumsum_energy >= rolloff_threshold)[0]
        spectral_rolloff = freqs[spectral_rolloff_idx[0]] if len(spectral_rolloff_idx) > 0 else fs / 2

        psd = np.sum(spec**2)
        spec_norm = spec / (np.sum(spec) + 1e-10)
        spectral_entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10))

        low_band_mask = freqs < 3.0
        mid_band_mask = (freqs >= 3.0) & (freqs < 8.0)
        high_band_mask = freqs >= 8.0

        energy_low = np.sum(spec[low_band_mask] ** 2) if np.any(low_band_mask) else 0
        energy_mid = np.sum(spec[mid_band_mask] ** 2) if np.any(mid_band_mask) else 0
        energy_high = np.sum(spec[high_band_mask] ** 2) if np.any(high_band_mask) else 0
        total_band_energy = energy_low + energy_mid + energy_high + 1e-10

        event_features = np.array(
            [
                duration_sec,
                start_time_min,
                mean_val,
                std_val,
                var_val,
                min_val,
                max_val,
                range_val,
                rms_val,
                skew_val,
                kurt_val,
                peak_pos,
                peak_neg,
                peak_overall,
                energy,
                crest_factor,
                shape_factor,
                impulse_factor,
                envelope_mean,
                envelope_max,
                envelope_std,
                zcr,
                p25,
                p50,
                p75,
                iqr,
                dom_freq,
                dom_freq_magnitude,
                mean_freq,
                spectral_centroid,
                spectral_std,
                spectral_rolloff,
                psd,
                spectral_entropy,
                energy_low / total_band_energy,
                energy_mid / total_band_energy,
                energy_high / total_band_energy,
            ]
        )

        features.append(event_features)
        event_info.append(
            {
                "event_id": event_id,
                "start_sample": start,
                "end_sample": end,
                "duration_sec": duration_sec,
                "start_time_min": start_time_min,
            }
        )

    return np.array(features), event_info


def apply_pca(features_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Reduce feature dimensionality for visualization."""
    pca = PCA(n_components=min(3, features_scaled.shape[1]))
    features_pca = pca.fit_transform(features_scaled)

    if features_pca.shape[1] < 3:
        padding = np.zeros((features_pca.shape[0], 3 - features_pca.shape[1]))
        features_pca = np.hstack([features_pca, padding])

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    return features_pca, pca


def run_kmeans_clustering(
    features_scaled: np.ndarray,
    n_clusters: int | None = None,
    k_max: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, int, list[int], list[float], list[float], KMeans]:
    """Run K-Means, auto-selecting k via silhouette score when not provided."""
    if len(features_scaled) < 2:
        raise ValueError("At least 2 events are required for clustering.")

    k_range = range(2, min(k_max + 1, len(features_scaled)))
    inertias: list[float] = []
    silhouette_scores: list[float] = []

    print("\nFinding optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores.append(sil_score)
        print(f"  k={k}: Silhouette Score={sil_score:.4f}, Inertia={kmeans.inertia_:.2f}")

    if n_clusters is None:
        optimal_k = list(k_range)[int(np.argmax(silhouette_scores))]
        print(f"Auto-selected k={optimal_k} (best silhouette score)")
    else:
        optimal_k = n_clusters
        print(f"Using user-specified k={optimal_k}")

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    labels = kmeans_final.fit_predict(features_scaled)

    print(f"\nK-Means evaluation (k={optimal_k}):")
    print(f"  Silhouette Score: {silhouette_score(features_scaled, labels):.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin_score(features_scaled, labels):.4f}")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz_score(features_scaled, labels):.2f}")

    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count} events ({percentage:.2f}%)")

    return labels, optimal_k, list(k_range), inertias, silhouette_scores, kmeans_final


def build_results_dataframe(
    event_info: list[dict],
    features: np.ndarray,
    kmeans_labels: np.ndarray,
    features_pca: np.ndarray,
) -> pd.DataFrame:
    """Combine event metadata, cluster labels, and key features."""
    results_df = pd.DataFrame(
        {
            "event_id": [info["event_id"] for info in event_info],
            "start_sample": [info["start_sample"] for info in event_info],
            "end_sample": [info["end_sample"] for info in event_info],
            "duration_sec": [info["duration_sec"] for info in event_info],
            "start_time_min": [info["start_time_min"] for info in event_info],
            "kmeans_cluster": kmeans_labels,
            "pca_1": features_pca[:, 0],
            "pca_2": features_pca[:, 1],
            "pca_3": features_pca[:, 2],
        }
    )
    results_df["peak_acceleration"] = features[:, 13]
    results_df["rms_acceleration"] = features[:, 8]
    results_df["energy"] = features[:, 14]
    results_df["dominant_freq"] = features[:, 26]
    results_df["spectral_centroid"] = features[:, 29]
    return results_df


def plot_clustering_results(
    k_range: list[int],
    inertias: list[float],
    silhouette_scores: list[float],
    optimal_k: int,
    features_pca: np.ndarray,
    kmeans_labels: np.ndarray,
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save K-Means clustering visualization."""
    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("K-Means Elbow Curve", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(k_range, silhouette_scores, "go-", linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color="r", linestyle="--", label=f"Selected k={optimal_k}")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    scatter1 = ax3.scatter(
        features_pca[:, 0],
        features_pca[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("K-Means Clustering (PCA)", fontweight="bold")
    plt.colorbar(scatter1, ax=ax3, label="Cluster")

    ax4 = plt.subplot(2, 3, 4, projection="3d")
    ax4.scatter(
        features_pca[:, 0],
        features_pca[:, 1],
        features_pca[:, 2],
        c=kmeans_labels,
        cmap="viridis",
        alpha=0.7,
        s=40,
    )
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.set_title("K-Means 3D View", fontweight="bold")

    ax5 = plt.subplot(2, 3, 5)
    scatter2 = ax5.scatter(
        results_df["duration_sec"],
        results_df["energy"],
        c=kmeans_labels,
        cmap="viridis",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax5.set_xlabel("Event Duration (s)")
    ax5.set_ylabel("Energy")
    ax5.set_title("Event Duration vs Energy (colored by cluster)", fontweight="bold")
    plt.colorbar(scatter2, ax=ax5, label="Cluster")
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(
        results_df["start_time_min"],
        kmeans_labels,
        c=kmeans_labels,
        cmap="viridis",
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    ax6.set_xlabel("Time (minutes)")
    ax6.set_ylabel("Cluster ID")
    ax6.set_title("Event Clusters Over Time", fontweight="bold")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Clustering visualization saved: {output_path}")


def run_event_detection(
    input_path: Path,
    output_dir: Path,
    sensor_column: str,
    delimiter: str = ";",
    fs: float = 100.0,
    low_hz: float = 0.2,
    high_hz: float = 16.0,
    threshold_percentile: float = 85.0,
    n_clusters: int | None = None,
) -> pd.DataFrame:


    """Run the full event detection and K-Means clustering pipeline."""
    date_hour = date_hour_folder_from_filename(input_path)
    output_dir = output_dir / f"{sensor_column}_{date_hour}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path, sensor_column, delimiter)
    acceleration_z = data["acceleration_z"].values.astype(float)

    signal_filtered, signal_raw = preprocess_signal(acceleration_z, fs, low_hz, high_hz)

    plot_preprocessing(
        signal_raw,
        signal_filtered,
        fs,
        low_hz,
        high_hz,
        output_dir / "preprocessing_steps.png",
    )

    signal = signal_filtered[:, 0]
    event_segments, threshold, envelope_smooth = detect_events(
        signal,
        fs,
        threshold_percentile=threshold_percentile,
    )

    plot_event_detection(
        signal,
        event_segments,
        envelope_smooth,
        threshold,
        threshold_percentile,
        fs,
        output_dir / "event_detection.png",
    )

    features, event_info = extract_event_features(event_segments, signal_filtered, fs)
    print(f"Feature matrix shape: {features.shape}")

    features_scaled = StandardScaler().fit_transform(features)
    features_pca, _ = apply_pca(features_scaled)

    kmeans_labels, optimal_k, k_range, inertias, silhouette_scores, _ = run_kmeans_clustering(
        features_scaled,
        n_clusters=n_clusters,
    )

    results_df = build_results_dataframe(event_info, features, kmeans_labels, features_pca)
    results_path = output_dir / "event_clustering_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    plot_clustering_results(
        k_range,
        inertias,
        silhouette_scores,
        optimal_k,
        features_pca,
        kmeans_labels,
        results_df,
        output_dir / "event_clustering_visualization.png",
    )

    print("\nAnalysis complete.")
    print(f"  Events analyzed: {len(features)}")
    print(f"  Features per event: {features.shape[1]}")
    print(f"  K-Means clusters: {optimal_k}")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect vehicle crossing events and cluster them with K-Means."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output CSV and plots",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default='030911EF_x',
        help="Sensor column name in the CSV (e.g. 030911EF_x)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=";",
        help="CSV delimiter (default: ;)",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Fixed number of K-Means clusters (default: auto-select via silhouette score)",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=85.0,
        help="Envelope threshold percentile for event detection (default: 85)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()



    run_event_detection(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        sensor_column=args.sensor,
        delimiter=args.delimiter,
        threshold_percentile=args.threshold_percentile,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
