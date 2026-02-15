import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional: Plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not installed. Install with 'pip install plotly' for interactive visualizations.")

TIME_BIN = "1min"
# Adjust based on your system - leave some cores for OS
NUM_WORKERS = max(1, cpu_count() - 200)


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

    for file_path in files:
        samples, missing, minute_data = process_single_file(file_path, sensor_name)
        
        total_samples += samples
        total_missing += missing

        for t, (miss, cnt) in minute_data.items():
            if t not in minute_accumulator:
                minute_accumulator[t] = [0, 0]
            minute_accumulator[t][0] += miss
            minute_accumulator[t][1] += cnt

    if total_samples == 0:
        return day, None, None

    daily_pct = 100 * total_missing / total_samples
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

    process_func = partial(process_day, root_dir=root_dir, sensor_name=sensor_name)

    daily_missing_pct = {}
    daily_time_missing = {}

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, day_folders),
            total=len(day_folders),
            desc="Processing days",
            unit="day"
        ))

    for day, pct, time_missing in results:
        if pct is not None:
            daily_missing_pct[day] = pct
            daily_time_missing[day] = time_missing

    return daily_missing_pct, daily_time_missing


# =========================
# VISUALIZATION 1: UPTIME/DOWNTIME TIMELINE
# =========================
def plot_uptime_timeline(daily_time_missing, output_dir):
    """
    Uptime/Downtime timeline showing sensor availability as horizontal bars.
    Green = working, Red = failed, White = no data.
    """
    days = sorted(daily_time_missing.keys())
    
    if not days:
        print("No data to plot for uptime timeline")
        return
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(days) * 0.15)))
    
    for idx, day in enumerate(days):
        time_data = daily_time_missing[day]
        
        # Get all times for this day
        times = sorted(time_data.keys())
        
        if not times:
            continue
        
        # Convert times to minutes since midnight for plotting
        for t in times:
            minutes = t.hour * 60 + t.minute
            missing_frac = time_data[t]
            
            # Color based on availability
            if missing_frac > 0.9:  # >90% missing = red (failed)
                color = '#d62728'
            elif missing_frac > 0.1:  # 10-90% missing = orange (degraded)
                color = '#ff7f0e'
            else:  # <10% missing = green (working)
                color = '#2ca02c'
            
            # Draw small rectangle for this minute
            ax.barh(idx, width=1, left=minutes, height=0.8, color=color, edgecolor='none')
    
    # Formatting
    ax.set_ylim(-0.5, len(days) - 0.5)
    ax.set_xlim(0, 24 * 60)
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days, fontsize=8)
    ax.set_xticks([h * 60 for h in range(0, 25, 3)])
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Date', fontsize=12)
    ax.set_title('Sensor Uptime/Downtime Timeline', fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ca02c', label='Working (<10% missing)'),
        mpatches.Patch(color='#ff7f0e', label='Degraded (10-90% missing)'),
        mpatches.Patch(color='#d62728', label='Failed (>90% missing)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uptime_timeline.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/uptime_timeline.png")
    plt.close()


def plot_uptime_timeline_interactive(daily_time_missing, output_dir):
    """
    Interactive Plotly version of uptime timeline with hover details.
    """
    if not PLOTLY_AVAILABLE:
        return
    
    days = sorted(daily_time_missing.keys())
    
    if not days:
        print("No data to plot for interactive uptime timeline")
        return
    
    fig = go.Figure()
    
    for idx, day in enumerate(days):
        time_data = daily_time_missing[day]
        times = sorted(time_data.keys())
        
        if not times:
            continue
        
        for t in times:
            minutes = t.hour * 60 + t.minute
            missing_frac = time_data[t]
            
            # Determine status and color
            if missing_frac > 0.9:
                color = '#d62728'
                status = 'Failed'
            elif missing_frac > 0.1:
                color = '#ff7f0e'
                status = 'Degraded'
            else:
                color = '#2ca02c'
                status = 'Working'
            
            # Add bar segment
            fig.add_trace(go.Bar(
                x=[1],
                y=[day],
                base=minutes,
                orientation='h',
                marker=dict(color=color),
                hovertemplate=f'<b>{day}</b><br>Time: {t.strftime("%H:%M")}<br>Status: {status}<br>Missing: {missing_frac*100:.1f}%<extra></extra>',
                showlegend=False
            ))
    
    # Add dummy traces for legend
    fig.add_trace(go.Bar(x=[0], y=[''], marker=dict(color='#2ca02c'), name='Working (<10%)', showlegend=True))
    fig.add_trace(go.Bar(x=[0], y=[''], marker=dict(color='#ff7f0e'), name='Degraded (10-90%)', showlegend=True))
    fig.add_trace(go.Bar(x=[0], y=[''], marker=dict(color='#d62728'), name='Failed (>90%)', showlegend=True))
    
    fig.update_layout(
        title='Sensor Uptime/Downtime Timeline (Hover for Details)',
        xaxis_title='Time of Day',
        yaxis_title='Date',
        barmode='stack',
        height=max(600, len(days) * 20),
        xaxis=dict(
            tickmode='array',
            tickvals=[h * 60 for h in range(0, 25, 3)],
            ticktext=[f'{h:02d}:00' for h in range(0, 25, 3)],
            range=[0, 24 * 60]
        ),
        hovermode='closest'
    )
    
    fig.write_html(f'{output_dir}/uptime_timeline_interactive.html')
    print(f"Saved: {output_dir}/uptime_timeline_interactive.html")


# =========================
# VISUALIZATION 2: OUTAGE SCATTER PLOT
# =========================
def plot_outage_scatter(daily_time_missing, output_dir, threshold=0.01):
    """
    Scatter plot showing ONLY failure events (>threshold missing).
    Reveals patterns in failure timing.
    """
    days = sorted(daily_time_missing.keys())
    
    if not days:
        print("No data to plot for outage scatter")
        return
    
    # Collect outage events
    outage_data = []
    
    for day_idx, day in enumerate(days):
        time_data = daily_time_missing[day]
        
        for t, missing_frac in time_data.items():
            if missing_frac > threshold:  # Only plot failures
                minutes = t.hour * 60 + t.minute
                outage_data.append({
                    'day_idx': day_idx,
                    'day': day,
                    'time': t,
                    'minutes': minutes,
                    'missing_frac': missing_frac
                })
    
    if not outage_data:
        print(f"No outages found above {threshold*100}% threshold")
        return
    
    df = pd.DataFrame(outage_data)
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(days) * 0.1)))
    
    # Scatter plot with size and color based on severity
    scatter = ax.scatter(
        df['minutes'],
        df['day_idx'],
        s=df['missing_frac'] * 100,  # Size = severity
        c=df['missing_frac'],
        cmap='Reds',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        vmin=threshold,
        vmax=1.0
    )
    
    # Formatting
    ax.set_xlim(0, 24 * 60)
    ax.set_ylim(-0.5, len(days) - 0.5)
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days, fontsize=8)
    ax.set_xticks([h * 60 for h in range(0, 25, 3)])
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Date', fontsize=12)
    ax.set_title(f'Outage Events (>{threshold*100}% Missing Data)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Fraction Missing')
    
    # Add grid
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/outage_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/outage_scatter.png")
    print(f"Total outage events plotted: {len(df)}")
    plt.close()


def plot_outage_scatter_interactive(daily_time_missing, output_dir, threshold=0.1):
    """
    Interactive Plotly scatter plot of outage events with hover details.
    """
    if not PLOTLY_AVAILABLE:
        return
    
    days = sorted(daily_time_missing.keys())
    
    if not days:
        print("No data to plot for interactive outage scatter")
        return
    
    outage_data = []
    
    for day_idx, day in enumerate(days):
        time_data = daily_time_missing[day]
        
        for t, missing_frac in time_data.items():
            if missing_frac > threshold:
                minutes = t.hour * 60 + t.minute
                outage_data.append({
                    'day': day,
                    'time_str': t.strftime('%H:%M'),
                    'minutes': minutes,
                    'missing_pct': missing_frac * 100,
                    'missing_frac': missing_frac
                })
    
    if not outage_data:
        print(f"No outages found above {threshold*100}% threshold")
        return
    
    df = pd.DataFrame(outage_data)
    
    fig = px.scatter(
        df,
        x='minutes',
        y='day',
        size='missing_pct',
        color='missing_frac',
        color_continuous_scale='Reds',
        hover_data={
            'day': True,
            'time_str': True,
            'missing_pct': ':.1f',
            'minutes': False,
            'missing_frac': False
        },
        labels={
            'minutes': 'Time of Day',
            'day': 'Date',
            'missing_frac': 'Fraction Missing',
            'time_str': 'Time',
            'missing_pct': 'Missing %'
        },
        title=f'Outage Events (>{threshold*100}% Missing Data) - Hover for Details'
    )
    
    # Update x-axis to show time
    fig.update_xaxes(
        tickmode='array',
        tickvals=[h * 60 for h in range(0, 25, 3)],
        ticktext=[f'{h:02d}:00' for h in range(0, 25, 3)],
        range=[0, 24 * 60]
    )
    
    fig.update_layout(
        height=max(600, len(days) * 15),
        yaxis={'categoryorder': 'category ascending'}
    )
    
    fig.write_html(f'{output_dir}/outage_scatter_interactive.html')
    print(f"Saved: {output_dir}/outage_scatter_interactive.html")




# =========================
# ENHANCED BAR CHART (Keep this one)
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


# =========================
# SUMMARY STATISTICS
# =========================
def generate_summary_stats(daily_missing_pct, daily_time_missing, output_dir):
    """Generate and save summary statistics."""
    stats = {
        'Total Days': len(daily_missing_pct),
        'Mean Missing %': np.mean(list(daily_missing_pct.values())),
        'Median Missing %': np.median(list(daily_missing_pct.values())),
        'Max Missing %': np.max(list(daily_missing_pct.values())),
        'Min Missing %': np.min(list(daily_missing_pct.values())),
        'Days > 5% missing': sum(1 for v in daily_missing_pct.values() if v > 5),
        'Days > 10% missing': sum(1 for v in daily_missing_pct.values() if v > 10),
    }
    
    # Count total outage events
    total_outages = 0
    for day_data in daily_time_missing.values():
        total_outages += sum(1 for frac in day_data.values() if frac > 0.1)
    stats['Total Outage Events (>10%)'] = total_outages
    
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
    
    # Visualization options
    GENERATE_INTERACTIVE = True  # Set to False to skip Plotly visualizations
    OUTAGE_THRESHOLD = 0.1  # 10% - adjust as needed
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting analysis...")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Sensor: {SENSOR_NAME}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Interactive plots: {'Enabled' if GENERATE_INTERACTIVE and PLOTLY_AVAILABLE else 'Disabled'}\n")
    
    # Run parallel analysis
    start_time = datetime.now()
    daily_missing_pct, daily_time_missing = analyze_missing_data_parallel(
        ROOT_DIR, SENSOR_NAME, NUM_WORKERS
    )
    end_time = datetime.now()
    
    print(f"\nAnalysis completed in {(end_time - start_time).total_seconds():.2f} seconds")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Daily percentage bar chart
    plot_missing_percentage(daily_missing_pct, OUTPUT_DIR)
    
    # 2. Uptime/Downtime Timeline
    print("Creating uptime timeline...")
    plot_uptime_timeline(daily_time_missing, OUTPUT_DIR)
    if GENERATE_INTERACTIVE and PLOTLY_AVAILABLE:
        plot_uptime_timeline_interactive(daily_time_missing, OUTPUT_DIR)
    
    # 3. Outage Scatter Plot
    print("Creating outage scatter plot...")
    plot_outage_scatter(daily_time_missing, OUTPUT_DIR, threshold=OUTAGE_THRESHOLD)
    if GENERATE_INTERACTIVE and PLOTLY_AVAILABLE:
        plot_outage_scatter_interactive(daily_time_missing, OUTPUT_DIR, threshold=OUTAGE_THRESHOLD)
    
    
    # Generate summary statistics
    generate_summary_stats(daily_missing_pct, daily_time_missing, OUTPUT_DIR)
    
    print("\n✓ All done!")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    if GENERATE_INTERACTIVE and PLOTLY_AVAILABLE:
        print("\nInteractive visualizations (open in browser):")
        print(f"  - {OUTPUT_DIR}/uptime_timeline_interactive.html")
        print(f"  - {OUTPUT_DIR}/outage_scatter_interactive.html")
        