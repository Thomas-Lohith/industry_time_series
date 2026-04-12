"""
Event-Based Unsupervised Detection for Bridge Monitoring
Uses vertical (Z-direction) acceleration data from CSV files
Implements: Event detection, K-Means, DBSCAN, Isolation Forest
Each data point = one vehicle crossing event
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ==================== Load Dataset ====================
print("Loading CSV datasets...")

csv_path1 = '/data/pool/c8x-98x/bridge_data/20250209/csv_acc/M001_2025-02-09_06-00-00_gg-84_int-7_th.csv'
csv_path2 = '/data/pool/c8x-98x/bridge_data/20250209/csv_acc/M001_2025-02-08_07-00-00_gg-83_int-8_th.csv'

# Load CSV files
try:
    data1 = pd.read_csv(csv_path1, sep=';')
    data1 = data1.drop(columns=data1.columns[data1.columns != '030911EF_x'])
    data1 = data1.dropna(subset=['030911EF_x'])
    #data1 = data1[:]
    print(f"✓ Loaded dataset1: {len(data1)} rows")

except FileNotFoundError:
    print("⚠ dataset1 not found, skipping...")
    data1 = pd.DataFrame()

try:
    data2 = pd.read_csv(csv_path2, sep=';')
    print(f"✓ Loaded dataset2: {len(data2)} rows")
except FileNotFoundError:
    print("⚠ dataset2 not found, skipping...")
    data2 = pd.DataFrame()

#data2 = pd.DataFrame()
# Combine datasets
if not data1.empty and not data2.empty:
    all_data = pd.concat([data1, data2], ignore_index=True)
elif not data1.empty:
    all_data = data1
elif not data2.empty:
    all_data = data2
else:
    raise FileNotFoundError("No CSV files found!")

print(f"Total rows loaded: {len(all_data)}")
print(f"Columns: {list(all_data.columns)}")

# ==================== Data Cleaning ====================
all_data['030911EF_x'] = all_data['030911EF_x'] - all_data['030911EF_x'].mean()

print(all_data.head(10))

# ==================== Data Validation ====================
print("\nValidating data...")

# Check if acceleration_z column exists
if '030911EF_x' not in all_data.columns:
    raise ValueError("CSV must contain '030911EF_x' column!")

# Remove rows with missing values
initial_count = len(all_data)
all_data = all_data.dropna(subset=['030911EF_x'])
dropped_count = initial_count - len(all_data)

if dropped_count > 0:
    print(f"⚠ Dropped {dropped_count} rows with missing values")

# Convert to numeric
all_data['acceleration_z'] = pd.to_numeric(all_data['030911EF_x'], errors='coerce')
all_data = all_data.dropna(subset=['acceleration_z'])

if len(all_data) == 0:
    raise ValueError("No valid acceleration_z data found!")

print(f"✓ Valid data points: {len(all_data)}")

# ==================== Preprocessing with Visualization ====================
print("\n" + "="*60)
print("SIGNAL PREPROCESSING - STEP BY STEP")
print("="*60)

# Extract vertical acceleration data
acceleration_z = all_data['acceleration_z'].values.astype(float)
acceleration = acceleration_z.reshape(-1, 1)

print(f"\n1. RAW SIGNAL")
print(f"   Shape: {acceleration.shape}")
print(f"   Range: [{acceleration.min():.4f}, {acceleration.max():.4f}]")
print(f"   Mean: {acceleration.mean():.4f}")
print(f"   Std: {acceleration.std():.4f}")

# Store for visualization
signal_raw = acceleration.copy()

# Apply Butterworth BAND-PASS filter (0.2-15 Hz)
print(f"\n2. APPLYING BUTTERWORTH BAND-PASS FILTER")
fs = 100  # Sampling frequency in Hz
low = 0.2   # High-pass cutoff (remove drift)
high = 16.0  # Low-pass cutoff (keep structural modes + vehicle frequencies)

b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')

# Ensure data is contiguous and proper dtype
acceleration = np.ascontiguousarray(acceleration, dtype=np.float64)

# Check for NaN or Inf values
if np.any(np.isnan(acceleration)) or np.any(np.isinf(acceleration)):
    print("   ⚠ Warning: NaN or Inf values found. Replacing with 0.")
    acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)

acceleration_filtered = filtfilt(b, a, acceleration, axis=0)

print(f"   Filter: {low} - {high} Hz band-pass")
print(f"   Range after filtering: [{acceleration_filtered.min():.4f}, {acceleration_filtered.max():.4f}]")
print(f"   Mean: {acceleration_filtered.mean():.4f}")
print(f"   Std: {acceleration_filtered.std():.4f}")

# Store for visualization
signal_filtered = acceleration_filtered.copy()

# Normalize data (standardization)
print(f"\n3. STANDARDIZATION (Z-SCORE NORMALIZATION)")
scaler_raw = StandardScaler()
acceleration_normalized = scaler_raw.fit_transform(acceleration) #can use filtered signal and raw signal here

print(f"   Range after scaling: [{acceleration_normalized.min():.4f}, {acceleration_normalized.max():.4f}]")
print(f"   Mean: {acceleration_normalized.mean():.4f}")
print(f"   Std: {acceleration_normalized.std():.4f}")

# Store for visualization
signal_normalized = acceleration_normalized.copy()

# ==================== VISUALIZATION: Signal Preprocessing Steps ====================
print("\n" + "="*60)
print("GENERATING PREPROCESSING VISUALIZATIONS")
print("="*60)

# Determine samples to plot (first 5000 or all if less)
n_samples_plot = min(90000, len(signal_raw))
time_axis = np.arange(n_samples_plot)

fig, axes = plt.subplots(4, 2, figsize=(20, 16))
fig.suptitle('Signal Preprocessing Pipeline - Vertical Acceleration (Z-direction)', 
             fontsize=16, fontweight='bold')

# Row 1: Raw Signal
ax1 = axes[0, 0]
ax1.plot(time_axis, signal_raw[:n_samples_plot, 0], linewidth=0.8, color='#1f77b4', alpha=0.7)
ax1.set_title('Step 1: Raw Signal', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f'Range: [{signal_raw.min():.3f}, {signal_raw.max():.3f}]\n'
                      f'Mean: {signal_raw.mean():.3f}\nStd: {signal_raw.std():.3f}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2 = axes[0, 1]
ax2.hist(signal_raw[:, 0], bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
ax2.set_title('Step 1: Raw Signal Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Acceleration (m/s²)')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3, axis='y')

# Row 2: Filtered Signal
ax3 = axes[1, 0]
ax3.plot(time_axis, signal_filtered[:n_samples_plot, 0], linewidth=0.8, color='#ff7f0e', alpha=0.7)
ax3.set_title('Step 2: After Butterworth Band-Pass Filter', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.98, f'Band: {low} - {high} Hz\n'
                      f'Range: [{signal_filtered.min():.3f}, {signal_filtered.max():.3f}]\n'
                      f'Mean: {signal_filtered.mean():.3f}\nStd: {signal_filtered.std():.3f}',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4 = axes[1, 1]
ax4.hist(signal_filtered[:, 0], bins=50, color='#ff7f0e', edgecolor='black', alpha=0.7)
ax4.set_title('Step 2: Filtered Signal Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Acceleration (m/s²)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3, axis='y')

# Row 3: Normalized Signal
ax5 = axes[2, 0]
ax5.plot(time_axis, signal_normalized[:n_samples_plot, 0], linewidth=0.8, color='#2ca02c', alpha=0.7)
ax5.set_title('Step 3: After Standardization', fontsize=12, fontweight='bold')
ax5.set_xlabel('Sample Index')
ax5.set_ylabel('Standardized Value')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax5.text(0.02, 0.98, f'Mean: {signal_normalized.mean():.3f} (≈0)\n'
                      f'Std: {signal_normalized.std():.3f} (≈1)\n'
                      f'Range: [{signal_normalized.min():.3f}, {signal_normalized.max():.3f}]',
         transform=ax5.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax6 = axes[2, 1]
ax6.hist(signal_normalized[:, 0], bins=50, color='#2ca02c', edgecolor='black', alpha=0.7)
ax6.set_title('Step 3: Standardized Distribution', fontsize=12, fontweight='bold')
ax6.set_xlabel('Standardized Value')
ax6.set_ylabel('Frequency')
ax6.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
ax6.grid(True, alpha=0.3, axis='y')

# Row 4: Comparison
ax7 = axes[3, 0]
# Normalize all signals to 0-1 for visual comparison
signal_raw_norm = (signal_raw[:n_samples_plot, 0] - signal_raw[:n_samples_plot, 0].min()) / \
                  (signal_raw[:n_samples_plot, 0].max() - signal_raw[:n_samples_plot, 0].min())
signal_filtered_norm = (signal_filtered[:n_samples_plot, 0] - signal_filtered[:n_samples_plot, 0].min()) / \
                       (signal_filtered[:n_samples_plot, 0].max() - signal_filtered[:n_samples_plot, 0].min())

ax7.plot(time_axis, signal_raw_norm, linewidth=1, alpha=0.5, label='Raw', color='#1f77b4')
ax7.plot(time_axis, signal_filtered_norm, linewidth=1.5, alpha=0.8, label='Filtered', color='#ff7f0e')
ax7.set_title('Comparison: Raw vs Filtered (Normalized for Visual)', fontsize=12, fontweight='bold')
ax7.set_xlabel('Sample Index')
ax7.set_ylabel('Normalized Amplitude [0-1]')
ax7.legend(loc='upper right')
ax7.grid(True, alpha=0.3)

ax8 = axes[3, 1]
# Power spectral density comparison
from scipy.signal import welch
f_raw, psd_raw = welch(signal_raw[:, 0], fs=fs, nperseg=2048, window='hann')
f_filt, psd_filt = welch(signal_filtered[:, 0], fs=fs, nperseg=2048, window='hann')
ax8.semilogy(f_raw, psd_raw, linewidth=1.5, alpha=0.7, label='Raw', color='#1f77b4')
ax8.semilogy(f_filt, psd_filt, linewidth=2, alpha=0.9, label='Filtered', color='#ff7f0e')
ax8.axvline(x=low, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'Low cutoff ({low} Hz)')
ax8.axvline(x=high, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'High cutoff ({high} Hz)')
ax8.set_title('Frequency Domain: Power Spectral Density', fontsize=12, fontweight='bold')
ax8.set_xlabel('Frequency (Hz)')
ax8.set_ylabel('PSD (dB/Hz)')
ax8.legend(loc='upper right', fontsize=8)
ax8.grid(True, alpha=0.3, which='both')
ax8.set_xlim([0, 25])  # Show up to 25 Hz

plt.tight_layout()
preprocessing_viz_file = 'preprocessing_steps.png'
plt.savefig(preprocessing_viz_file, dpi=150, bbox_inches='tight')
print(f"✓ Preprocessing visualization saved: {preprocessing_viz_file}")
plt.close()

# ==================== Event Detection ====================
print("\n" + "="*60)
print("EVENT DETECTION (Vehicle Passage Detection)")
print("="*60)

# Compute envelope of filtered signal
signal = acceleration[:, 0]#can use normalised signal and raw signal here
envelope = np.abs(signal)

# Smooth envelope
smooth_sec = 0.5  # 1 second smoothing window
smooth_N = max(1, int(fs * smooth_sec))
envelope_smooth = uniform_filter1d(envelope, size=smooth_N)

# Threshold based on upper percentile of activity
threshold_percentile = 85  # Adjust this if needed (higher = fewer events)
thr = np.percentile(envelope_smooth, threshold_percentile)
is_active = envelope_smooth > thr

print(f"Envelope threshold (p{threshold_percentile}): {thr:.4f}")
print(f"Active samples: {np.sum(is_active)} / {len(is_active)} ({100*np.sum(is_active)/len(is_active):.2f}%)")

# Parameters for event detection
min_event_sec = 1  # Minimum event duration (expect 20-30s)
merge_gap_sec = 1.5  # Merge events closer than this
min_event_N = int(min_event_sec * fs)
merge_gap_N = int(merge_gap_sec * fs)

# Find continuous active segments
events = []
start = None
for i, active in enumerate(is_active):
    if active and start is None:
        start = i
    elif not active and start is not None:
        events.append((start, i))
        start = None
if start is not None:
    events.append((start, len(is_active)))

print(f"\nInitial segments found: {len(events)}")

# Merge close segments
merged = []
for s, e in events:
    if not merged:
        merged.append([s, e])
    else:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_N:
            merged[-1][1] = e
        else:
            merged.append([s, e])

# Filter by minimum duration
event_segments = [(s, e) for s, e in merged if (e - s) >= min_event_N]

print(f"After merging (gap < {merge_gap_sec}s): {len(merged)} segments")
print(f"After filtering (duration >= {min_event_sec}s): {len(event_segments)} events")

if len(event_segments) == 0:
    print("\n⚠ WARNING: No events detected!")
    print("   Try adjusting threshold_percentile (lower value) or min_event_sec")
    raise ValueError("No events detected with current parameters")

# Display event statistics
durations = [(e - s) / fs for s, e in event_segments]
print(f"\nEvent duration statistics:")
print(f"  Min: {min(durations):.2f} s")
print(f"  Max: {max(durations):.2f} s")
print(f"  Mean: {np.mean(durations):.2f} s")
print(f"  Median: {np.median(durations):.2f} s")

# Visualize event detection
fig, axes = plt.subplots(3, 1, figsize=(20, 12))
fig.suptitle('Event Detection - Vehicle Passage Identification', fontsize=16, fontweight='bold')

# Plot 1: Filtered signal with detected events
ax1 = axes[0]
time_samples = np.arange(len(signal)) / fs / 60  # Convert to minutes
ax1.plot(time_samples, signal, linewidth=0.5, alpha=0.6, label='Filtered Signal')
for i, (start, end) in enumerate(event_segments):
    ax1.axvspan(start/fs/60, end/fs/60, alpha=0.3, color='red', label='Detected Event' if i == 0 else '')
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Normalized Acceleration')
ax1.set_title('Filtered Signal with Detected Events', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Envelope and threshold
ax2 = axes[1]
ax2.plot(time_samples, envelope_smooth, linewidth=1, label='Smoothed Envelope', color='blue')
ax2.axhline(y=thr, color='red', linestyle='--', linewidth=2, label=f'Threshold (p{threshold_percentile})')
for i, (start, end) in enumerate(event_segments):
    ax2.axvspan(start/fs/60, end/fs/60, alpha=0.2, color='green')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Envelope Amplitude')
ax2.set_title('Envelope and Detection Threshold', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Event durations
ax3 = axes[2]
event_ids = np.arange(len(event_segments))
event_starts = [s/fs/60 for s, e in event_segments]
ax3.bar(event_ids, durations, color='skyblue', edgecolor='black')
ax3.axhline(y=np.mean(durations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(durations):.1f}s')
ax3.set_xlabel('Event ID')
ax3.set_ylabel('Duration (seconds)')
ax3.set_title(f'Event Durations (Total: {len(event_segments)} events)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
event_detection_file = 'event_detection.png'
plt.savefig(event_detection_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Event detection visualization saved: {event_detection_file}")
plt.close()

#==================== Feature Extraction Per Event ====================
print("\n" + "="*60)
print("FEATURE EXTRACTION (Per Event)")
print("="*60)

features = []
event_start_indices = []
event_info = []  # Store event metadata

for event_id, (start, end) in enumerate(event_segments):
    seg = acceleration_normalized[start:end, 0]  # 1D segment of this event
    
    # Event metadata
    duration_sec = (end - start) / fs
    start_time_min = start / fs / 60
    
    # Time-domain Statistical Features
    mean_val = np.mean(seg)
    std_val = np.std(seg)
    var_val = np.var(seg)
    min_val = np.min(seg)
    max_val = np.max(seg)
    range_val = max_val - min_val
    rms_val = np.sqrt(np.mean(seg**2))
    skew_val = skew(seg)
    kurt_val = kurtosis(seg)
    
    # Peak metrics
    peak_pos = np.max(seg)
    peak_neg = np.abs(np.min(seg))
    peak_overall = np.max(np.abs(seg))
    
    # Energy
    energy = np.sum(seg**2)
    
    # Shape factors
    crest_factor = peak_overall / (rms_val + 1e-10)
    shape_factor = rms_val / (np.mean(np.abs(seg)) + 1e-10)
    impulse_factor = peak_overall / (np.mean(np.abs(seg)) + 1e-10)
    
    # Envelope features
    envelope_mean = np.mean(np.abs(seg))
    envelope_max = np.max(np.abs(seg))
    envelope_std = np.std(np.abs(seg))
    
    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(seg)) != 0)
    zcr = zero_crossings / len(seg)
    
    # Percentiles
    p25 = np.percentile(seg, 25)
    p50 = np.percentile(seg, 50)
    p75 = np.percentile(seg, 75)
    iqr = p75 - p25
    
    # Frequency-domain Features
    spec = np.abs(fft(seg))
    spec = spec[:len(spec)//2]  # Keep only positive frequencies
    freqs = np.linspace(0, fs/2, len(spec))
    
    # Dominant frequency
    dom_freq_idx = np.argmax(spec)
    dom_freq = freqs[dom_freq_idx]
    dom_freq_magnitude = spec[dom_freq_idx]
    
    # Spectral statistics
    mean_freq = np.mean(spec)
    spectral_centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-10)
    spectral_std = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * spec) / (np.sum(spec) + 1e-10))
    
    # Spectral rolloff (85% energy)
    cumsum_energy = np.cumsum(spec**2)
    total_energy = cumsum_energy[-1]
    rolloff_threshold = 0.85 * total_energy
    spectral_rolloff_idx = np.where(cumsum_energy >= rolloff_threshold)[0]
    spectral_rolloff = freqs[spectral_rolloff_idx[0]] if len(spectral_rolloff_idx) > 0 else fs/2
    
    # Power spectral density
    psd = np.sum(spec**2)
    
    # Spectral entropy
    spec_norm = spec / (np.sum(spec) + 1e-10)
    spectral_entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10))
    
    # Frequency band energies (useful for different vehicle types)
    # Low: 0-3 Hz, Mid: 3-8 Hz, High: 8-15 Hz
    low_band_mask = freqs < 3.0
    mid_band_mask = (freqs >= 3.0) & (freqs < 8.0)
    high_band_mask = freqs >= 8.0
    
    energy_low = np.sum(spec[low_band_mask]**2) if np.any(low_band_mask) else 0
    energy_mid = np.sum(spec[mid_band_mask]**2) if np.any(mid_band_mask) else 0
    energy_high = np.sum(spec[high_band_mask]**2) if np.any(high_band_mask) else 0
    
    # Normalize band energies
    total_band_energy = energy_low + energy_mid + energy_high + 1e-10
    energy_low_ratio = energy_low / total_band_energy
    energy_mid_ratio = energy_mid / total_band_energy
    energy_high_ratio = energy_high / total_band_energy
    
    # Combine all features
    event_features = np.array([
        # Event metadata (2)
        duration_sec, start_time_min,
        
        # Basic statistics (9)
        mean_val, std_val, var_val, min_val, max_val, 
        range_val, rms_val, skew_val, kurt_val,
        
        # Peak metrics (3)
        peak_pos, peak_neg, peak_overall,
        
        # Energy and shape (4)
        energy, crest_factor, shape_factor, impulse_factor,
        
        # Envelope (3)
        envelope_mean, envelope_max, envelope_std,
        
        # Zero-crossing (1)
        zcr,
        
        # Distribution (4)
        p25, p50, p75, iqr,
        
        # Frequency metrics (8)
        dom_freq, dom_freq_magnitude, mean_freq,
        spectral_centroid, spectral_std, spectral_rolloff,
        psd, spectral_entropy,
        
        # Frequency band energies (3)
        energy_low_ratio, energy_mid_ratio, energy_high_ratio
    ])
    
    features.append(event_features)
    event_start_indices.append(start)
    event_info.append({
        'event_id': event_id,
        'start_sample': start,
        'end_sample': end,
        'duration_sec': duration_sec,
        'start_time_min': start_time_min
    })

features = np.array(features)
print(f"Feature matrix shape: {features.shape}")
print(f"Number of features per event: {features.shape[1]}")
print(f"Total events analyzed: {len(features)}")

# Feature names for reference
feature_names = [
    'duration_sec', 'start_time_min',
    'mean', 'std', 'var', 'min', 'max', 'range', 'rms', 'skew', 'kurtosis',
    'peak_pos', 'peak_neg', 'peak_overall',
    'energy', 'crest_factor', 'shape_factor', 'impulse_factor',
    'envelope_mean', 'envelope_max', 'envelope_std',
    'zcr',
    'p25', 'p50', 'p75', 'iqr',
    'dom_freq', 'dom_freq_mag', 'mean_freq', 
    'spectral_centroid', 'spectral_std', 'spectral_rolloff',
    'psd', 'spectral_entropy',
    'energy_low', 'energy_mid', 'energy_high'
]

# Standardize features for clustering
scaler_features = StandardScaler()
features_scaled = scaler_features.fit_transform(features)

# ==================== Dimensionality Reduction ====================
print("\nApplying PCA for visualization...")
pca = PCA(n_components=min(2, features_scaled.shape[1]))
features_pca = pca.fit_transform(features_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

# Ensure we have 3 components for visualization
if features_pca.shape[1] < 3:
    padding = np.zeros((features_pca.shape[0], 3 - features_pca.shape[1]))
    features_pca = np.hstack([features_pca, padding])


# ==================== Clustering - K-Means ====================
print("\n" + "="*60)
print("K-MEANS CLUSTERING")
print("="*60)

# Find optimal number of clusters
inertias = []
silhouette_scores = []
k_range = range(2, min(11, len(features)))  # Ensure k < n_samples

print("\nFinding optimal number of clusters...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(features_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"  k={k}: Silhouette Score={sil_score:.4f}, Inertia={kmeans.inertia_:.2f}")

# Choose optimal k
optimal_k = 6          #          k_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal number of clusters: {optimal_k}")

# Final K-Means clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(features_scaled)

# Evaluation metrics
kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(features_scaled, kmeans_labels)
kmeans_ch = calinski_harabasz_score(features_scaled, kmeans_labels)

print(f"\nK-Means Evaluation (k={optimal_k}):")
print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  Davies-Bouldin Index: {kmeans_db:.4f}")
print(f"  Calinski-Harabasz Score: {kmeans_ch:.2f}")

# Cluster distribution
unique, counts = np.unique(kmeans_labels, return_counts=True)
print(f"\nCluster Distribution:")
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(kmeans_labels)) * 100
    print(f"  Cluster {cluster_id}: {count} events ({percentage:.2f}%)")





# ==================== Clustering - DBSCAN ====================
print("\n" + "="*60)
print("DBSCAN CLUSTERING")
print("="*60)

# DBSCAN
dbscan = DBSCAN(eps=0.015, min_samples=max(2, len(features)*2))  # Adaptive min_samples
dbscan_labels = dbscan.fit_predict(features_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Number of clusters: {n_clusters_dbscan}")
print(f"  Number of noise events: {n_noise} ({(n_noise/len(dbscan_labels)*100):.2f}%)")

if n_clusters_dbscan > 1:
    mask = dbscan_labels != -1
    if np.sum(mask) > 0:
        dbscan_silhouette = silhouette_score(features_scaled[mask], dbscan_labels[mask])
        print(f"  Silhouette Score: {dbscan_silhouette:.4f}")

# Cluster distribution
unique, counts = np.unique(dbscan_labels, return_counts=True)
print(f"\nCluster Distribution:")
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(dbscan_labels)) * 100
    if cluster_id == -1:
        print(f"  Noise: {count} events ({percentage:.2f}%)")
    else:
        print(f"  Cluster {cluster_id}: {count} events ({percentage:.2f}%)")

# ==================== Anomaly Detection - Isolation Forest ====================
print("\n" + "="*60)
print("ANOMALY DETECTION (Isolation Forest)")
print("="*60)

iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(features_scaled)
anomaly_scores = iso_forest.score_samples(features_scaled)

# Convert labels
anomaly_binary = (anomaly_labels == -1).astype(int)

n_anomalies = np.sum(anomaly_binary)
n_normal = len(anomaly_binary) - n_anomalies

print(f"\nIsolation Forest Results:")
print(f"  Normal events: {n_normal} ({(n_normal/len(anomaly_binary)*100):.2f}%)")
print(f"  Anomalies detected: {n_anomalies} ({(n_anomalies/len(anomaly_binary)*100):.2f}%)")

# ==================== Results Summary ====================
print("\n" + "="*60)
print("SUMMARY AND RECOMMENDATIONS")
print("="*60)

# Create results dataframe
results_df = pd.DataFrame({
    'event_id': [info['event_id'] for info in event_info],
    'start_sample': [info['start_sample'] for info in event_info],
    'end_sample': [info['end_sample'] for info in event_info],
    'duration_sec': [info['duration_sec'] for info in event_info],
    'start_time_min': [info['start_time_min'] for info in event_info],
    'kmeans_cluster': kmeans_labels,
    'dbscan_cluster': dbscan_labels,
    'is_anomaly': anomaly_binary,
    'anomaly_score': anomaly_scores,
    'pca_1': features_pca[:, 0],
    'pca_2': features_pca[:, 1],
    'pca_3': features_pca[:, 2]
})

# Add key features
results_df['peak_acceleration'] = features[:, 13]  # peak_overall
results_df['rms_acceleration'] = features[:, 8]     # rms
results_df['energy'] = features[:, 14]              # energy
results_df['dominant_freq'] = features[:, 26]       # dom_freq
results_df['spectral_centroid'] = features[:, 29]   # spectral_centroid

print("\nEvent Detection Summary:")
print(f"  Total events analyzed: {len(features)}")
print(f"  Features per event: {features.shape[1]}")
print(f"  K-Means identified {optimal_k} distinct patterns")
print(f"  DBSCAN identified {n_clusters_dbscan} clusters + {n_noise} outliers")
print(f"  Isolation Forest detected {n_anomalies} anomalies")

print("\nInterpretation:")
print("  • Each data point = one vehicle crossing event")
print("  • Clusters likely represent different vehicle types/loads/speeds")
print("  • Anomalies may indicate unusual events (heavy loads, sudden braking, etc.)")

# Save results
output_file = 'event_clustering_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# ==================== Visualization ====================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Elbow curve
ax1 = plt.subplot(3, 3, 1)
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('K-Means Elbow Curve', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Silhouette scores
ax2 = plt.subplot(3, 3, 2)
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs k', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. K-Means clusters (PCA 2D)
ax3 = plt.subplot(3, 3, 3)
scatter1 = ax3.scatter(features_pca[:, 0], features_pca[:, 1], 
                       c=kmeans_labels, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_title('K-Means Clustering (PCA)', fontweight='bold')
plt.colorbar(scatter1, ax=ax3, label='Cluster')

# 4. K-Means clusters (PCA 3D)
ax4 = plt.subplot(3, 3, 4, projection='3d')
scatter2 = ax4.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2],
                       c=kmeans_labels, cmap='viridis', alpha=0.7, s=40)
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')
ax4.set_title('K-Means 3D View', fontweight='bold')

# 5. DBSCAN clusters
ax5 = plt.subplot(3, 3, 5)
scatter3 = ax5.scatter(features_pca[:, 0], features_pca[:, 1],
                       c=dbscan_labels, cmap='plasma', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
ax5.set_title('DBSCAN Clustering', fontweight='bold')
plt.colorbar(scatter3, ax=ax5, label='Cluster')

# 6. Anomaly detection
ax6 = plt.subplot(3, 3, 6)
colors = ['green' if x == 0 else 'red' for x in anomaly_binary]
ax6.scatter(features_pca[:, 0], features_pca[:, 1],
            c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax6.set_xlabel('PC1')
ax6.set_ylabel('PC2')
ax6.set_title('Anomaly Detection', fontweight='bold')
# Create legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Normal'),
                   Patch(facecolor='red', label='Anomaly')]
ax6.legend(handles=legend_elements, loc='upper right')

# 7. Event duration vs energy
ax7 = plt.subplot(3, 3, 7)
scatter4 = ax7.scatter(results_df['duration_sec'], results_df['energy'],
                       c=kmeans_labels, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax7.set_xlabel('Event Duration (s)')
ax7.set_ylabel('Energy')
ax7.set_title('Event Duration vs Energy (colored by cluster)', fontweight='bold')
plt.colorbar(scatter4, ax=ax7, label='Cluster')
ax7.grid(True, alpha=0.3)

# 8. Dominant frequency distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(results_df['dominant_freq'], bins=30, color='coral', edgecolor='black', alpha=0.7)
ax8.set_xlabel('Dominant Frequency (Hz)')
ax8.set_ylabel('Number of Events')
ax8.set_title('Dominant Frequency Distribution', fontweight='bold')
ax8.axvline(x=np.mean(results_df['dominant_freq']), color='r', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(results_df["dominant_freq"]):.2f} Hz')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. Cluster assignment timeline
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(results_df['start_time_min'], kmeans_labels, 
            c=kmeans_labels, cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax9.set_xlabel('Time (minutes)')
ax9.set_ylabel('Cluster ID')
ax9.set_title('Event Clusters Over Time', fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = 'event_clustering_visualization.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {plot_file}")
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print(f"  1. {preprocessing_viz_file} - Signal preprocessing steps")
print(f"  2. {event_detection_file} - Event detection visualization")
print(f"  3. {plot_file} - Clustering and analysis results")
print(f"  4. {output_file} - Detailed event-level results")
print("\nNext steps:")
print("  - Review clusters to identify vehicle patterns")
print("  - Investigate anomalies for unusual crossings")
print("  - Correlate with video/traffic counter if available")
print("  - Use cluster labels for supervised learning")
print("="*60)