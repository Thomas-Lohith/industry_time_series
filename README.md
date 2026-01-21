# Bridge Structural Health Monitoring & Traffic Analysis

A comprehensive Python-based repository for analyzing bridge vibration data and correlating it with traffic patterns. This project combines accelerometer sensor data processing with vehicle transit analysis to assess bridge structural health and dynamic loading patterns.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Usage](#usage)
- [Scripts Documentation](#scripts-documentation)
- [Analysis Workflows](#analysis-workflows)
- [Contributing](#contributing)

## 🎯 Overview

This repository provides tools for:
- **Vibration Analysis**: Processing accelerometer data from 106 bridge-mounted sensors
- **Traffic Pattern Analysis**: Analyzing vehicle counts, classifications, and lane distributions
- **Structural Load Assessment**: Correlating heavy vehicle traffic with bridge vibrations
- **Frequency Domain Analysis**: Spectrograms, FFT, and power spectral density (PSD) computations
- **Data Filtering**: Threshold-based event detection and DC component removal

## ✨ Features

### Vibration Data Processing
- Multi-sensor time-series visualization
- DC offset removal and signal preprocessing
- Spectrogram generation (2D and 3D)
- Power spectral density (Welch method)
- FFT analysis with peak detection
- Threshold-based event filtering

### Traffic Analysis
- Vehicle classification (Car, Bus, Truck, Van, Motorbike)
- Lane distribution analysis
- Heavy vehicle pattern identification
- Weekday vs. weekend traffic comparison
- Peak hour identification
- Moving average smoothing

### Advanced Features
- Memory-efficient data loading with Polars
- Batch processing of large datasets
- Frequency reduction for long-term monitoring
- Cross-validation between multiple data sources
- Automated threshold computation

## 🔧 Installation

### Prerequisites

```bash
Python 3.8+
```

### Dependencies

```bash
pip install pandas polars numpy scipy matplotlib plotly
pip install psutil argparse openpyxl pyarrow
```

### Optional Tools
```bash
pip install jupyter  # For notebook examples
```

## 📊 Data Sources

The repository works with three primary data types:

1. **Accelerometer Data** (CSV/Parquet)
   - 106 sensors across the bridge
   - 100 Hz sampling rate
   - X, Y, Z acceleration components
   - Format: `YYYYMMDD/csv_acc/M001_YYYY-MM-DD_HH-00-00_gg-*_int-*_th.csv`

2. **Traffic Data** (Excel)
   - **Classi**: Vehicle type classification (5 categories)
   - **Corsie**: Lane distribution (4 lanes + emergency)
   - **Transiti**: Total vehicle counts (15-minute intervals)

3. **Dynamic Weight Data** (Excel)
   - Individual vehicle measurements
   - Axle configurations and spacing
   - Speed and gross weight estimates

## 🚀 Usage

### Quick Start Examples

#### 1. Analyze Vibration Data for Specific Time Window

```bash
python3 vibration_analysis.py \
  --path /path/to/data.csv \
  --start_time "2025/03/07 01:05:00" \
  --duration_mins 5 \
  --sensor "030911FF_x"
```

#### 2. Generate Power Spectral Density

```bash
python3 spectogram.py \
  --path /path/to/data.csv \
  --sensor "030911EF_x" \
  --date 20250303
```

#### 3. Compare Multiple Datasets

```bash
python3 spectrum.py \
  --paths file1.csv file2.csv file3.csv \
  --sensor "030911D2_x"
```

#### 4. Analyze Traffic Patterns

```bash
python3 traffic.py \
  --path /path/to/transiti.xlsx \
  --date "01/03/2025"
```

#### 5. Filter Events by Threshold

```bash
python3 filtering_script_v5.py \
  --file_path /path/to/data.parquet \
  --chunk 100 \
  --thresholds 0.0005,0.001,0.002 \
  --sensor "0309101E_x"
```

## 📚 Scripts Documentation

### Core Analysis Scripts

#### `vibration_analysis.py` / `Precise_vibration_analysis.py`
Comprehensive vibration data analysis with visualization capabilities.

**Key Features:**
- Multi-sensor time-series plotting
- Spectrogram generation
- Histogram distributions
- Memory-efficient Polars-based loading
- Support for both CSV and Parquet formats

**Arguments:**
- `--path`: Path to data file (CSV/Parquet)
- `--start_time`: Start timestamp (format: YYYY/MM/DD HH:MM:SS)
- `--duration_mins`: Duration in minutes
- `--sensor`: Sensor ID(s) to analyze (comma-separated)
- `--backend`: Visualization backend (matplotlib/plotly)
- `--output_dir`: Directory for output graphs

**Example:**
```bash
python3 Precise_vibration_analysis.py \
  --path data/20250307/csv_acc/M001_2025-03-07_01-00-00.csv \
  --start_time "2025/03/07 01:05:00" \
  --duration_mins 5 \
  --sensor "030911FF_x,03091017_z" \
  --backend matplotlib \
  --output_dir ./graphs
```

---

#### `spectogram.py`
Frequency domain analysis with multiple visualization options.

**Features:**
- 2D and 3D spectrograms
- FFT spectrum analysis
- Power spectral density (Welch method)
- Window comparison (Hann, Hamming, Blackman, Flattop)
- Peak frequency detection

**Arguments:**
- `--path`: Path to data file
- `--sensor`: Sensor column name
- `--date`: Target date (for file naming)

**Output:**
- Spectrogram plots
- FFT spectrum graphs
- PSD comparison with peak annotations

---

#### `spectrum.py`
Compare power spectral densities across multiple datasets.

**Use Case:** Analyze vibration patterns during different traffic conditions.

**Arguments:**
- `--paths`: Multiple file paths (space-separated)
- `--sensor`: Sensor column to analyze
- `--date`: Optional target date

**Example:**
```bash
python3 spectrum.py \
  --paths data/morning.csv data/evening.csv \
  --sensor "030911D2_x"
```

---

#### `Traffic_analysis.py`
Comprehensive traffic pattern analysis with structural load assessment.

**Features:**
- Daily traffic volume trends
- Vehicle type distribution
- Heavy vehicle patterns (trucks + buses)
- Lane usage statistics
- Weekday vs. weekend comparison
- Structural load index calculation

**Output:**
- 4 comprehensive visualization figures
- Statistical summaries
- Peak hour identification

**Example:**
```python
python3 Traffic_analysis.py
# Data paths are configured in the script
```

---

#### `traffic.py`
Moving average analysis for transit data.

**Features:**
- Time-based moving averages
- Multiple time window comparisons (30, 60, 120 min)
- Daily pattern visualization

**Arguments:**
- `--path`: Path to Excel file
- `--date`: Target date (DD/MM/YYYY)

---

### Data Processing Scripts

#### `filtering_script_v5.py`
Advanced threshold-based event detection.

**Features:**
- DC offset removal
- Threshold-based filtering with windowing
- RMSE calculation for filter quality
- Signal filtering ratio metrics
- Batch threshold testing

**Arguments:**
- `--file_path`: Path to data file
- `--chunk`: Chunk size (samples to extend after threshold crossing)
- `--thresholds`: Comma-separated threshold values
- `--sensor`: Sensor column name

**Example:**
```bash
python3 filtering_script_v5.py \
  --file_path data.parquet \
  --chunk 100 \
  --thresholds 0.0005,0.001,0.0015,0.002 \
  --sensor "0309101E_x"
```

---

#### `frequency_reduction.py`
Reduce data frequency for long-term storage and analysis.

**Features:**
- Compute mean and variance over chunks
- Log-normal variance transform
- Process entire date folders
- Daily output files

**Arguments:**
- `--root_dir`: Parent folder with date subfolders
- `--sensor_channel`: Sensor column name
- `--chunk_size`: Samples per averaging chunk (default: 100)
- `--output`: Output directory

**Example:**
```bash
python3 frequency_reduction.py \
  --root_dir /data/sensors/ \
  --sensor_channel "03091002_x" \
  --chunk_size 100 \
  --output ./reduced_data/
```

---

#### `thresholds.py`
Automatically compute detection thresholds for all sensors.

**Formula:** `threshold = mean + (x × std)`

**Arguments:**
- `--input_csv`: Input CSV file
- `--output_csv`: Output CSV with thresholds
- `--x`: Multiplier for standard deviation (default: 3)

**Example:**
```bash
python3 thresholds.py \
  --input_csv data.csv \
  --output_csv sensor_thresholds.csv \
  --x 3
```

---

### Utility Scripts

#### `less_traffice_event_finder.py`
Identify time periods with minimal traffic for controlled vibration analysis.

**Arguments:**
- `--path`: Path to Excel file with timestamps
- `--time_difference`: Minimum gap in minutes

**Output:** Text file with isolated event timestamps

---

#### `graph_for_etfa.py`
Generate publication-ready graphs for conference presentations.

**Features:**
- Clean, formatted time-series plots
- Threshold visualization overlay
- Custom styling for academic publications

---

## 🔄 Analysis Workflows

### Workflow 1: Correlate Traffic with Vibrations

```bash
# Step 1: Analyze traffic patterns
python3 Traffic_analysis.py

# Step 2: Identify peak heavy vehicle hours (e.g., 6-10 AM)

# Step 3: Extract vibration data for those periods
python3 vibration_analysis.py \
  --path data/20250307/csv_acc/M001_2025-03-07_06-00-00.csv \
  --start_time "2025/03/07 06:00:00" \
  --duration_mins 240 \
  --sensor "030911FF_x"

# Step 4: Compare with low-traffic periods
python3 spectrum.py \
  --paths heavy_traffic.csv light_traffic.csv \
  --sensor "030911FF_x"
```

### Workflow 2: Structural Health Assessment

```bash
# Step 1: Compute baseline thresholds
python3 thresholds.py \
  --input_csv baseline_data.csv \
  --output_csv thresholds.csv \
  --x 3

# Step 2: Filter anomalous events
python3 filtering_script_v5.py \
  --file_path current_data.parquet \
  --chunk 100 \
  --thresholds 0.001,0.002 \
  --sensor "0309101E_x"

# Step 3: Frequency analysis of detected events
python3 spectogram.py \
  --path filtered_events.csv \
  --sensor "0309101E_x" \
  --date 20250315
```

### Workflow 3: Long-Term Monitoring

```bash
# Step 1: Reduce data frequency
python3 frequency_reduction.py \
  --root_dir /data/march_2025/ \
  --sensor_channel "03091002_x" \
  --chunk_size 100 \
  --output ./monthly_summary/

# Step 2: Analyze trends over time
# Use reduced data for monthly/yearly comparisons
```

## 📁 Project Structure

```bash
├── analysis
│   ├── Precise_vibration_analysis.py
│   ├── Traffic_analysis.py
│   ├── __init__.py
│   ├── frequency_reduction.py
│   ├── less_traffice_event_finder.py
│   ├── spectogram.py
│   ├── spectrum.py
│   ├── traffic.py
│   └── vibration_analysis.py
├── data_format_conversion
│   ├── __init__.py
│   └── csv_to_parquet.py
├── dataset
│   ├── Febbraio
│   │   ├── classi.xlsx
│   │   ├── corsie.xlsx
│   │   ├── transiti.xlsx
│   │   ├── ~$classi.xlsx
│   │   ├── ~$corsie.xlsx
│   │   └── ~$transiti.xlsx
│   └── __init__.py
├── filtering
│   ├── __init__.py
│   ├── filtering_script_v4.py
│   ├── filtering_script_v5.py
│   ├── filtering_wt_polars.py
│   ├── graph_for_etfa.py
│   └── thresholds.py
├── research
│   ├── MSE_downsampling_rates.ipynb
│   ├── eda.ipynb
│   ├── helper.ipynb
│   └── traffic_data_single_file.ipynb
├── results
│   ├── events_lesstraffic.txt
│   ├── filtering
│   │   ├── ETFA.png
│   │   ├── histogram_0309101E_x.png
│   │   └── rmse_graph_wip.png
│   ├── graphs
│   │   ├── 0309100F_x_superimposed_psd.png
│   │   └── windows_comp
│   │       └── psd_window_comparison_with_peak_detection_20freq{start_time}.png
│   ├── old_thresholds_abs.csv
│   ├── sensors.csv
│   ├── thresholds.csv
│   ├── thresholds_abs.csv
│   └── vibration_data_sensors.html
└── shared
    ├── __init__.py
    ├── sensor_data.py
    └── sensor_map.py
```

## 🔬 Sensor Configuration

The bridge has **106 accelerometer sensors** distributed across multiple spans:

- **Campate 1A**: Sensors 104-106 (e.g., `030911FF_x`, `030911EF_x`, `03091200_x`)
- **Campate 1B**: Sensors 51-53 (e.g., `0309100F_x`, `030910F6_x`, `0309101E_x`)
- **Campate 2**: Sensors 99-101 (e.g., `030911D2_x`, `03091005_x`, `0309101F_x`)

Each sensor measures:
- **X-axis**: Vertical direction (primary for bridge deflection)
- **Y-axis**: Flexural direction (lateral movement)
- **Z-axis**: Torsional direction (rotational effects)

## 📊 Key Analysis Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling Rate | 100 Hz | Accelerometer data collection frequency |
| Traffic Intervals | 15 minutes | Vehicle count aggregation period |
| Typical Chunk Size | 100 samples | 1 second of data at 100 Hz |
| Spectrogram Window | 256-2048 | FFT window size (nperseg) |
| Heavy Vehicles | Trucks + Buses | Primary structural load contributors |

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Real-time monitoring dashboard
- [ ] Machine learning anomaly detection
- [ ] Automated report generation
- [ ] Integration with weather data
- [ ] Modal analysis for natural frequencies
- [ ] Damage detection algorithms

## 📝 Citation

If you use this code in your research, please cite:

```
Bridge Structural Health Monitoring Repository
University of Pavia Industrial PhD Program
2025
```

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Contact the research team at [university email]

## ⚠️ Important Notes

1. **Data Format**: Ensure CSV files use semicolon (`;`) as delimiter
2. **Time Format**: Use `YYYY/MM/DD HH:MM:SS:fff` for timestamps
3. **Memory**: Large files (>1GB) should use Polars-based scripts
4. **Sensor Names**: Must match exactly (case-sensitive)

## 🔍 Troubleshooting

**Issue**: `KeyError: 'time'`
- **Solution**: Check that time column exists and matches expected format

**Issue**: High memory usage
- **Solution**: Use Parquet format and Polars-based scripts

**Issue**: Empty visualizations
- **Solution**: Verify start_time and duration_mins cover data range

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**License**: MIT (specify as needed)