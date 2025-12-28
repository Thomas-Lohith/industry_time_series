
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import psutil
import os
from scipy import signal
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Union, List, Tuple, Optional
import logging

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants (centralized, not hardcoded)
CONFIG = {
    'SAMPLING_RATE': 100,  # Hz
    'SPECTROGRAM_NPERSEG': 256,
    'SPECTROGRAM_NOVERLAP': 64,
    'SPECTROGRAM_MAX_FREQ': 50,  # Hz
    'SUBPLOT_COLS': 3,
    'HISTOGRAM_BINS': 50,
    'DPI': 300,
    'TIME_FORMAT': '%Y/%m/%d %H:%M:%S:%f',
}

# Default sensor definitions (can be overridden via CLI)
SENSOR_PRESETS = {
    'campate2': ['030911D2_x', '03091005_x', '0309101F_x'],
    'campate1b': ['0309100F_x', '030910F6_x', '0309101E_x'],
    'campate1a': ['030911FF_x', '030911EF_x', '03091200_x', '03091155_z', '03091207_x', '03091119_z'],
    'all_campate': ['030911FF_x', '03091017_z', '03091113_x', '0309123B_z', '03091111_z', '03091003_x'],
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def memory_usage(label: str = "") -> float:
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    msg = f"Memory usage: {mem_mb:.2f} MB"
    if label:
        msg = f"[{label}] {msg}"
    logger.info(msg)
    return mem_mb


def calculate_grid_size(n_items: int, cols: int = 3) -> Tuple[int, int]:
    """
    Calculate grid dimensions for subplots.
    
    Args:
        n_items: Number of items to plot
        cols: Number of columns desired
        
    Returns:
        Tuple of (rows, cols)
    """
    rows = -(-n_items // cols)  # Ceiling division
    return rows, cols


def parse_sensor_ids(sensor_str: Optional[str], available_sensors: List[str]) -> List[str]:
    if sensor_str is None:
        logger.info(f"No sensor specified, using all {len(available_sensors)} sensors")
        return available_sensors
    
    # Parse comma-separated values
    requested = [s.strip() for s in sensor_str.split(',')]
    
    # Validate each sensor exists
    invalid = [s for s in requested if s not in available_sensors]
    if invalid:
        raise ValueError(
            f"Invalid sensor IDs: {invalid}\n"
            f"Available sensors: {available_sensors}"
        )
    logger.info(f"Using {len(requested)} sensor(s): {requested}")
    return requested


def validate_inputs(args: argparse.Namespace) -> None:

    # Check file exists
    if not Path(args.path).exists():
        raise FileNotFoundError(f"Data file not found: {args.path}")
    
    # Validate duration
    if args.duration_mins <= 0:
        raise ValueError(f"Duration must be positive, got: {args.duration_mins}")
    
    logger.info(f"Input validation passed")


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data_polars(filepath: str) -> Tuple[pl.DataFrame, List[str], str]:
    """
    Load data efficiently using Polars.
    
    Args:
        filepath: Path to CSV or Parquet file
        
    Returns:
        Tuple of (DataFrame, sensor_columns, time_column_name)
        
    Raises:
        ValueError: If file format not supported
    """
    logger.info(f"Loading data from: {filepath}")
    memory_usage("before_load")
    
    # Determine file type and load
    if filepath.endswith('.csv'):
        df = pl.read_csv(filepath, separator=';')
        is_lazy = False
    elif filepath.endswith('.parquet'):
        df = pl.scan_parquet(filepath)
        is_lazy = True
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Get schema
    schema = df.collect_schema() if is_lazy else df.schema
    total_rows = (
        df.select(pl.len()).collect().item() 
        if is_lazy 
        else df.select(pl.count()).item()
    )
    
    logger.info(f"Total rows: {total_rows:,}")
    
    # Identify time column
    time_candidates = ['time', 'timestamp', 'date']
    time_column = next(
        (col for col in schema.keys() if col.lower() in time_candidates),
        None
    )
    if not time_column:
        raise ValueError(f"No time column found. Available: {list(schema.keys())}")
    
    # Get sensor columns (all except time)
    sensor_columns = [col for col in schema.keys() if col != time_column]
    logger.info(f"Found {len(sensor_columns)} sensor columns")
    logger.info(f"Time column: {time_column}")
    
    memory_usage("after_load")
    
    # Collect if lazy
    if is_lazy:
        df = df.collect()
    
    return df, sensor_columns, time_column


def filter_dc_by_mean(
    df: pl.DataFrame, 
    sensor_columns: List[str]
) -> pl.DataFrame:
   
    logger.info("Removing DC component from sensor data...")
    memory_usage("before_dc_filter")
    
    # Use lazy evaluation for efficiency
    result_df = df.with_columns([
        (pl.col(col) - pl.col(col).mean()).alias(col)
        for col in sensor_columns
    ])
    
    memory_usage("after_dc_filter")
    return result_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def calculate_subplot_layout(n_sensors: int, cols: int = None) -> Tuple[int, int]:
    """
    Calculate subplot grid dimensions.
    
    Args:
        n_sensors: Number of sensors
        cols: Preferred number of columns (default: CONFIG['SUBPLOT_COLS'])
        
    Returns:
        (rows, cols)
    """
    cols = cols or CONFIG['SUBPLOT_COLS']
    rows = -(-n_sensors // cols)  # Ceiling division
    return rows, cols


def visualize_single_sensor(
    df: pd.DataFrame,
    sensor_id: str,
    time_column: str,
    backend: str = 'matplotlib',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize a single sensor's time-series data.
    
    Args:
        df: DataFrame with time and sensor columns
        sensor_id: Sensor column name to plot
        time_column: Name of time column
        backend: 'matplotlib' or 'plotly'
        output_path: Path to save figure (if None, no save)
        figsize: Figure size for matplotlib
    """
    if sensor_id not in df.columns:
        raise ValueError(f"Sensor '{sensor_id}' not found in data")
    
    logger.info(f"Visualizing single sensor: {sensor_id} [{backend}]")
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df[time_column], df[sensor_id], linewidth=1.5, color='steelblue')
        ax.set_title(f"Sensor: {sensor_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Acceleration", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=CONFIG['DPI'])
            logger.info(f"Saved to: {output_path}")
        plt.show()
        
    elif backend == 'plotly':
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[time_column],
                y=df[sensor_id],
                mode='lines',
                name=sensor_id,
                line=dict(width=1.5, color='steelblue')
            )
        )
        fig.update_layout(
            title=f"Sensor: {sensor_id}",
            xaxis_title="Time",
            yaxis_title="Acceleration",
            hovermode='x unified',
            template='plotly_white'
        )
        fig.show()
        
        if output_path and output_path.endswith('.html'):
            fig.write_html(output_path)
            logger.info(f"Saved to: {output_path}")


def visualize_multiple_sensors(
    df: pd.DataFrame,
    sensor_ids: List[str],
    time_column: str,
    backend: str = 'matplotlib',
    output_path: Optional[str] = None,
    cols: int = None
) -> None:
    """
    Visualize multiple sensors as subplots.
    
    Args:
        df: DataFrame with time and sensor columns
        sensor_ids: List of sensor column names to plot
        time_column: Name of time column
        backend: 'matplotlib' or 'plotly'
        output_path: Path to save figure
        cols: Number of subplot columns
    """
    invalid = [s for s in sensor_ids if s not in df.columns]
    if invalid:
        raise ValueError(f"Sensors not found: {invalid}")
    
    logger.info(f"Visualizing {len(sensor_ids)} sensors [{backend}]")
    n_sensors = len(sensor_ids)
    rows, cols = calculate_subplot_layout(n_sensors, cols)
    
    if backend == 'matplotlib':
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharex=True)
        axes = np.atleast_1d(axes).flatten()  # Handle single row/col case
        
        for i, sensor_id in enumerate(sensor_ids):
            ax = axes[i]
            ax.plot(df[time_column], df[sensor_id], linewidth=1, color='steelblue')
            ax.set_title(sensor_id, fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(n_sensors, len(axes)):
            fig.delaxes(axes[j])
        
        fig.suptitle("Sensor Vibration Analysis", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_path:
            fig.savefig(output_path, dpi=CONFIG['DPI'])
            logger.info(f"Saved to: {output_path}")
        plt.show()
        
    elif backend == 'plotly':
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=sensor_ids,
            specs=[[{"secondary_y": False}] * cols for _ in range(rows)]
        )
        
        for i, sensor_id in enumerate(sensor_ids):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=df[time_column],
                    y=df[sensor_id],
                    mode='lines',
                    name=sensor_id,
                    line=dict(width=1),
                    opacity=0.8
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title_text="Sensor Vibration Analysis",
            showlegend=False,
            height=300 * rows,
            template='plotly_white'
        )
        fig.show()
        
        if output_path and output_path.endswith('.html'):
            fig.write_html(output_path)
            logger.info(f"Saved to: {output_path}")


def visualize_sensors(
    df: pd.DataFrame,
    sensor_ids: Union[str, List[str]],
    time_column: str,
    backend: str = 'matplotlib',
    output_path: Optional[str] = None,
    cols: int = None
) -> None:
    """
    Unified visualization function for single or multiple sensors.
    Automatically routes to appropriate visualization function.
    
    Args:
        df: DataFrame with time and sensor columns
        sensor_ids: Single sensor ID (str) or list of sensor IDs
        time_column: Name of time column
        backend: 'matplotlib' or 'plotly'
        output_path: Path to save figure
        cols: Number of subplot columns (for multiple sensors)
    """
    # Convert single sensor to list for consistent handling
    if isinstance(sensor_ids, str):
        visualize_single_sensor(df, sensor_ids, time_column, backend, output_path)
    else:
        visualize_multiple_sensors(df, sensor_ids, time_column, backend, output_path, cols)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def multi_sensor_spectrogram(
    df: pd.DataFrame,
    sensor_ids: List[str],
    cols: int = None,
    output_path: Optional[str] = None
) -> None:
    """
    Plot spectrograms for multiple sensors in subplots.
    
    Args:
        df: DataFrame with sensor data
        sensor_ids: List of sensor column names
        cols: Number of subplot columns
        output_path: Path to save figure
    """
    logger.info(f"Creating spectrograms for {len(sensor_ids)} sensors...")
    
    cols = cols or CONFIG['SUBPLOT_COLS']
    rows, cols = calculate_subplot_layout(len(sensor_ids), cols)
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5*cols, 4*rows),
        sharex=False
    )
    axes = np.atleast_1d(axes).flatten()
    
    for i, sensor_id in enumerate(sensor_ids):
        x = df[sensor_id].values
        f, t, Sxx = signal.spectrogram(
            x,
            fs=CONFIG['SAMPLING_RATE'],
            window=signal.get_window('hamming', CONFIG['SPECTROGRAM_NPERSEG']),
            nperseg=CONFIG['SPECTROGRAM_NPERSEG'],
            noverlap=CONFIG['SPECTROGRAM_NOVERLAP']
        )
        
        # Convert to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        
        ax = axes[i]
        pcm = ax.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='viridis')
        ax.set_title(sensor_id, fontsize=10, fontweight='bold')
        ax.set_ylabel('Freq [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_ylim([0, CONFIG['SPECTROGRAM_MAX_FREQ']])
        fig.colorbar(pcm, ax=ax, label='Power [dB]')
    
    # Remove empty subplots
    for j in range(len(sensor_ids), len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle('Spectrograms of Selected Sensors', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        fig.savefig(output_path, dpi=CONFIG['DPI'])
        logger.info(f"Saved to: {output_path}")
    plt.show()


def visualize_sensor_histograms(
    df: pd.DataFrame,
    sensor_ids: List[str],
    bins: int = None,
    output_path: Optional[str] = None
) -> None:
    """
    Create histograms for sensor data distribution analysis.
    
    Args:
        df: DataFrame with sensor data
        sensor_ids: List of sensor column names
        bins: Number of histogram bins
        output_path: Path to save figure
    """
    logger.info(f"Creating histograms for {len(sensor_ids)} sensors...")
    bins = bins or CONFIG['HISTOGRAM_BINS']
    
    grid_size = int(np.ceil(np.sqrt(len(sensor_ids))))
    fig = plt.figure(figsize=(4*grid_size, 4*grid_size))
    
    for i, sensor_id in enumerate(sensor_ids, 1):
        ax = plt.subplot(grid_size, grid_size, i)
        ax.hist(df[sensor_id].dropna(), bins=bins, alpha=0.7, color='steelblue')
        ax.set_title(sensor_id, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=CONFIG['DPI'])
        logger.info(f"Saved to: {output_path}")
    plt.show()


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main analysis workflow."""
    
    # Setup CLI arguments
    parser = argparse.ArgumentParser(
        description='Analyze vibration data from bridge sensors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sensor analysis
  python3 script.py --path data.csv --start_time "2025/03/07 01:05:00" --duration_mins 5 --sensor "030911FF_x"
  
  # Multiple sensors
  python3 script.py --path data.csv --start_time "2025/03/07 01:05:00" --duration_mins 5 --sensor "030911FF_x,03091017_z,03091113_x"
  
  # All available sensors
  python3 script.py --path data.csv --start_time "2025/03/07 01:05:00" --duration_mins 5
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to CSV or Parquet data file'
    )
    parser.add_argument(
        '--start_time',
        type=str,
        required=True,
        help='Start time (format: YYYY/MM/DD HH:MM:SS)'
    )
    parser.add_argument(
        '--duration_mins',
        type=float,
        required=True,
        help='Duration in minutes to analyze'
    )
    parser.add_argument(
        '--sensor',
        type=str,
        default=None,
        help='Sensor ID(s) to analyze (comma-separated for multiple)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['matplotlib', 'plotly'],
        default='matplotlib',
        help='Visualization backend'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./graphs',
        help='Directory to save output figures'
    )
    parser.add_argument(
        '--no-spectrogram',
        action='store_true',
        help='Skip spectrogram generation'
    )
    parser.add_argument(
        '--no-histogram',
        action='store_true',
        help='Skip histogram generation'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data
        df, available_sensors, time_column = load_data_polars(args.path)
        
        # Parse sensor IDs
        sensor_ids = parse_sensor_ids(args.sensor, available_sensors)
        
        # Remove DC component
        df = filter_dc_by_mean(df, available_sensors)
        
        # Convert to pandas and handle time window
        logger.info("Converting to pandas and filtering time window...")
        df_pd = (
            df.select([time_column] + sensor_ids)
            .to_pandas()
        )
        
        df_pd[time_column] = pd.to_datetime(
            df_pd[time_column],
            format=CONFIG['TIME_FORMAT'],
            errors='coerce'
        )
        
        # Filter to time window
        start_time = pd.to_datetime(args.start_time)
        end_time = start_time + pd.Timedelta(minutes=args.duration_mins)
        
        df_pd = df_pd[
            (df_pd[time_column] >= start_time) &
            (df_pd[time_column] <= end_time)
        ]
        
        logger.info(f"Data window: {df_pd[time_column].min()} to {df_pd[time_column].max()}")
        logger.info(f"Records in window: {len(df_pd):,}")
        
        # === TASK 2: Visualization ===
        # Main time-series visualization
        output_file = Path(args.output_dir) / f"vibration_{start_time.strftime('%Y%m%d_%H%M%S')}.png"
        if args.backend == 'plotly':
            output_file = output_file.with_suffix('.html')
        
        visualize_sensors(
            df_pd,
            sensor_ids,
            time_column,
            backend=args.backend,
            output_path=str(output_file)
        )
        
        # Spectrogram analysis
        if not args.no_spectrogram:
            spec_file = Path(args.output_dir) / f"spectrogram_{start_time.strftime('%Y%m%d_%H%M%S')}.png"
            multi_sensor_spectrogram(
                df_pd,
                sensor_ids,
                output_path=str(spec_file)
            )
        
        # Histogram analysis
        if not args.no_histogram:
            hist_file = Path(args.output_dir) / f"histogram_{start_time.strftime('%Y%m%d_%H%M%S')}.png"
            visualize_sensor_histograms(
                df_pd,
                sensor_ids,
                output_path=str(hist_file)
            )
        
        memory_usage("final")
        logger.info("Analysis complete!")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
