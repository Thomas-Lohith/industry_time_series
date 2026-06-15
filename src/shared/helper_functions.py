def find_csv_file(root_folder, date_str, hour):
    """Locate the CSV file for a given date and hour inside root_folder."""
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    csv_folder = os.path.join(root_folder, date_str, date_str, "csv_acc")
    for fname in os.listdir(csv_folder):
        if (fname.startswith(f"M001_{formatted_date}_{hour:02d}-00-00_")
                and f"_int-{hour+1}_" in fname and fname.endswith("_th.csv")):
            return os.path.join(csv_folder, fname)
    raise FileNotFoundError(f"No matching CSV file found for {date_str} hour {hour}")


def load_data_polars(filepath):
    """Load parquet file using Polars for memory efficiency"""
    print("Loading data with Polars...")
    ()
    # Load the data
    
    check1 = filepath.endswith('.csv')
    check2 = filepath.endswith('.parquet')
    if check1:
        df = pl.read_csv(filepath, separator= ';')
        print('reading the csv file:')
        # Count total rows without loading everything to memory
        total_rows = df.select(pl.len()).item() #use .collect if we use scan_csv function
        print(f"Total rows: {total_rows:,}")
    
        # Get column names for sensor data (assuming pattern ends with _z for vertical direction)
        sensor_columns = [col for col in df.columns if col != 'time']
        
        # Try to identify time column
        time_column_candidates = ['time', 'timestamp', 'date']
        time_column = next((col for col in df.columns if col in time_column_candidates), None)
        print(f"Found {len(sensor_columns)} sensor columns:")
        print(f"Using '{time_column}' as the time column")
    if check2:
        df = pl.scan_parquet(filepath)    
        # Count total rows without loading everything to memory
        total_rows = df.select(pl.len()).collect().item()
        print(f"Total rows: {total_rows:,}")
        
        # Get column names for sensor data (assuming pattern ends with _z for vertical direction)
        sensor_columns = [col for col in df.collect_schema().keys() if col != 'time']
        
        # Try to identify time column
        time_column_candidates = ['time', 'timestamp', 'date']
        time_column = next((col for col in df.collect_schema().keys() if col.lower() in time_column_candidates), None)
        
        print(f"Found {len(sensor_columns)} sensor columns:")
        print(f"Using '{time_column}' as the time column")
    
    
    return df, sensor_columns, time_column

def parse_sensor_ids(sensor_str, available_sensors):
    if sensor_str is None:
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
    return requested

# =========================================================
# THRESHOLD EXTRACTION
# =========================================================
def extract_sensor_thresholds(bridge, sensor_ids):
    """Extract sensor-specific trigger thresholds from bridge model."""
    sensor_thresholds = {}
    
    for sensor_id in sensor_ids:
        sensor = bridge[sensor_id]  # Sensor object
        sensor_thresholds[sensor_id] = sensor.trigger_threshold
    
    return sensor_thresholds
