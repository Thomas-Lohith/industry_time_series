import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

TIME_BIN = "1min"
# Adjust based on your system - leave some cores for OS
NUM_WORKERS = max(1, cpu_count() - 20)


# =========================
# OPTIMIZED CORE ANALYSIS
# =========================
def process_single_file(file_path, sensor_name):
    """
    Process a single CSV file efficiently.
    Returns: (total_samples, total_missing, minute_data)
    """
    try:
        # Use chunking for large files to reduce memory footprint
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(
            file_path,
            sep=";",
            usecols=["time", sensor_name],
            dtype={sensor_name: "float32"},
            chunksize=chunk_size
        ):
            chunks.append(chunk)
        
        if not chunks:
            return 0, 0, {}
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Parse time more efficiently
        df["time"] = pd.to_datetime(
            df["time"], 
            format='%Y/%m/%d %H:%M:%S:%f', 
            errors="coerce"
        )
        df = df.dropna(subset=["time"])

        if df.empty:
            return 0, 0, {}

        missing_mask = df[sensor_name].isna()
        total_samples = len(df)
        total_missing = missing_mask.sum()

        # Minute binning - vectorized approach
        time_bins = df["time"].dt.floor(TIME_BIN)
        
        # Group by minute and count missing
        minute_data = {}
        for t in time_bins.unique():
            mask = time_bins == t
            key = t.time()
            missing_count = missing_mask[mask].sum()
            total_count = mask.sum()
            minute_data[key] = (int(missing_count), int(total_count))

        return total_samples, total_missing, minute_data

    except Exception as e:
        # Silently skip problematic files or log if needed
        return 0, 0, {}


def process_day(day, root_dir, sensor_name):
    """
    Process all files for a single day.
    Returns: (day, daily_pct, daily_time_missing)
    """
    day_path = os.path.join(root_dir, day)
    csv_acc_path = os.path.join(day_path, "csv_acc")

    if not os.path.exists(csv_acc_path):
        return day, None, None

    total_samples = 0
    total_missing = 0
    minute_accumulator = {}

    files = [
        os.path.join(csv_acc_path, f)
        for f in sorted(os.listdir(csv_acc_path))
        if f.endswith(".csv")
    ]

    # Process files sequentially within day (can also parallelize this if needed)
    for file_path in files:
        samples, missing, minute_data = process_single_file(file_path, sensor_name)
        
        total_samples += samples
        total_missing += missing

        # Merge minute data
        for t, (miss, cnt) in minute_data.items():
            if t not in minute_accumulator:
                minute_accumulator[t] = [0, 0]
            minute_accumulator[t][0] += miss
            minute_accumulator[t][1] += cnt

    if total_samples == 0:
        return day, None, None

    # Calculate daily missing percentage
    daily_pct = 100 * total_missing / total_samples

    # Calculate minute-level missing fraction
    daily_time_missing = {
        t: miss / cnt
        for t, (miss, cnt) in minute_accumulator.items()
    }

    return day, daily_pct, daily_time_missing


def analyze_missing_data_parallel(root_dir, sensor_name, num_workers=NUM_WORKERS):
    """
    Parallel version of missing data analysis.
    Processes multiple days simultaneously.
    """
    day_folders = [
        d for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    print(f"Processing {len(day_folders)} days using {num_workers} workers...")

    # Create partial function with fixed parameters
    process_func = partial(process_day, root_dir=root_dir, sensor_name=sensor_name)

    daily_missing_pct = {}
    daily_time_missing = {}

    # Parallel processing with progress bar
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, day_folders),
            total=len(day_folders),
            desc="Processing days",
            unit="day"
        ))

    # Aggregate results
    for day, pct, time_missing in results:
        if pct is not None:
            daily_missing_pct[day] = pct
            daily_time_missing[day] = time_missing

    return daily_missing_pct, daily_time_missing


# =========================
# IMPROVED VISUALIZATION
# =========================
def plot_missing_percentage(daily_missing_pct, output_dir):
    """Enhanced bar plot with better styling."""
    plt.figure(figsize=(14, 5))
    
    days = list(daily_missing_pct.keys())
    values = list(daily_missing_pct.values())
    
    colors = ['#d62728' if v > 10 else '#ff7f0e' if v > 5 else '#2ca02c' 
              for v in values]
    
    plt.bar(days, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    # plt.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    # plt.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("% Missing Data", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.title(f"Daily Missing Data Percentage", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/missing_data_percent.png', dpi=300)
    print(f"Saved: {output_dir}/missing_data_percent.png")
    plt.close()


def plot_missing_heatmap_improved(daily_time_missing, output_dir):
    """
    IMPROVED HEATMAP with proper time-of-day labels and better visualization.
    """
    # Create DataFrame with days as rows and time-of-day as columns
    heatmap_df = pd.DataFrame(daily_time_missing).T.sort_index()
    
    # Sort columns by time
    heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)
    
    # Convert time objects to strings for better labels
    time_labels = [t.strftime('%H:%M') for t in heatmap_df.columns]
    
    # Reduce number of labels to avoid crowding (show every Nth label)
    n_labels = len(time_labels)
    step = max(1, n_labels // 24)  # Show ~24 labels max
    
    # Create reduced labels list
    reduced_labels = [time_labels[i] if i % step == 0 else '' for i in range(n_labels)]
    
    plt.figure(figsize=(16, 10))
    
    # Use seaborn for better heatmap
    ax = sns.heatmap(
        heatmap_df,
        cmap='RdYlGn_r',  # Red = high missing, Green = low missing
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Fraction Missing'},
        xticklabels=reduced_labels,  # Pass the reduced labels directly
        yticklabels=True,
        linewidths=0,
        rasterized=True  # For large plots
    )
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.xlabel("Time of Day", fontsize=12)
    plt.ylabel("Date", fontsize=12)
    plt.title("Missing Data Heatmap (Day × Time-of-Day)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_improved.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/heatmap_improved.png")
    plt.close()


def plot_missing_probability_improved(daily_time_missing, output_dir):
    """Enhanced time-of-day missing probability plot."""
    heatmap_df = pd.DataFrame(daily_time_missing)
    missing_prob = heatmap_df.mean(axis=1).sort_index()
    
    # Convert time to minutes for better plotting
    times = [t.hour * 60 + t.minute for t in missing_prob.index]
    
    plt.figure(figsize=(14, 5))
    plt.plot(times, missing_prob.values, linewidth=2, color='#1f77b4')
    plt.fill_between(times, missing_prob.values, alpha=0.3)
    
    # Add horizontal reference line
    mean_missing = missing_prob.mean()
    plt.axhline(y=mean_missing, color='red', linestyle='--', 
                alpha=0.5, label=f'Mean: {mean_missing:.2%}')
    
    # Format x-axis as time
    hours = range(0, 25, 3)
    plt.xticks([h * 60 for h in hours], [f'{h:02d}:00' for h in hours])
    
    plt.ylabel("Missing Probability", fontsize=12)
    plt.xlabel("Time of Day", fontsize=12)
    plt.title("Average Missing Data Probability Throughout the Day", 
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/missing_probability.png', dpi=300)
    print(f"Saved: {output_dir}/missing_probability.png")
    plt.close()


def generate_summary_stats(daily_missing_pct, daily_time_missing, output_dir):
    """Generate and save summary statistics."""
    stats = {
        'Total Days': len(daily_missing_pct),
        'Mean Missing %': np.mean(list(daily_missing_pct.values())),
        'Median Missing %': np.median(list(daily_missing_pct.values())),
        'Max Missing %': np.max(list(daily_missing_pct.values())),
        'Min Missing %': np.min(list(daily_missing_pct.values())),
        'Days > 10% missing': sum(1 for v in daily_missing_pct.values() if v > 10),
        'Days > 20% missing': sum(1 for v in daily_missing_pct.values() if v > 20),
    }
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.2f}")
        else:
            print(f"{key:.<40} {value}")
    print("="*50 + "\n")
    
    # Save to file
    with open(f'{output_dir}/summary_stats.txt', 'w') as f:
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Saved: {output_dir}/summary_stats.txt")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    
    ROOT_DIR = "/data/pool/c8x-98x/bridge_data"
    SENSOR_NAME = "03091002_x"
    OUTPUT_DIR = '/home/c8x-98x/industry_time_series/src/results/missing_stats'
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting analysis...")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Sensor: {SENSOR_NAME}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Run parallel analysis
    start_time = datetime.now()
    daily_missing_pct, daily_time_missing = analyze_missing_data_parallel(
        ROOT_DIR, SENSOR_NAME, NUM_WORKERS
    )
    end_time = datetime.now()
    
    print(f"\nAnalysis completed in {(end_time - start_time).total_seconds():.2f} seconds")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_missing_percentage(daily_missing_pct, OUTPUT_DIR)
    plot_missing_heatmap_improved(daily_time_missing, OUTPUT_DIR)
    plot_missing_probability_improved(daily_time_missing, OUTPUT_DIR)
    
    # Generate summary statistics
    generate_summary_stats(daily_missing_pct, daily_time_missing, OUTPUT_DIR)
    
    print("\n✓ All done!")