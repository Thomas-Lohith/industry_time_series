import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def process_sensor_files(root_dir, sensor_name, output_dir, chunk_size=100):
    """
    Reduce frequency of sensor data by computing mean and variance every 'chunk_size' samples.
    Applies log-normal transform on variance.
    Saves one reduced CSV per day into the specified output directory.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each day folder
    for day_folder in sorted(os.listdir(root_dir)):
        day_path = os.path.join(root_dir, day_folder)
        if not os.path.isdir(day_path):
            continue

        csv_acc_path = os.path.join(day_path, "csv_acc")
        if not os.path.exists(csv_acc_path):
            print(f"Skipping {day_folder}: no csv_acc folder found")
            continue

        print(f"\nProcessing day: {day_folder}")
        results = []

        # Process hourly CSVs inside csv_acc folder
        for file in tqdm(sorted(os.listdir(csv_acc_path)), desc=f"{day_folder}", unit="file"):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(csv_acc_path, file)

            try:
                df = pd.read_csv(file_path, usecols=["time", sensor_name], sep=';')
            except ValueError:
                print(f"Skipping file (missing columns): {file_path}, check the column names!")
                continue

            # Compute reduced features in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    continue  # skip incomplete chunk at end

                mean_val = chunk[sensor_name].mean()
                var_val = chunk[sensor_name].var()
                log_var = log_normal_variance(var_val)

                results.append({
                    "day": day_folder,
                    "hour_file": file,
                    "start_time": chunk["time"].iloc[0],
                    "end_time": chunk["time"].iloc[-1],
                    "mean": mean_val,
                    "variance": var_val,
                    "log_variance": log_var,
                })

        # Save the day's result to a CSV file
        if results:
            results_df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f"{day_folder}.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Saved reduced CSV for {day_folder} → {output_path} ({len(results_df)} rows)")
        else:
            print(f" No valid data found for {day_folder}")


def log_normal_variance(variance):
    """Apply log-normal transform safely to variance values."""
    if variance <= 0 or pd.isna(variance):
        return np.nan
    return np.log(variance)


def main():
    parser = argparse.ArgumentParser(description="Reduce frequency of sensor CSV data (daily outputs)")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to parent folder containing date folders")
    parser.add_argument("--sensor_channel", type=str, required=True, help="Sensor column name (e.g., 03091002_x)")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of samples per averaging chunk")
    parser.add_argument("--output", type=str, required=True, help="Directory to save reduced daily CSVs")

    args = parser.parse_args()
    process_sensor_files(args.root_dir, args.sensor_channel, args.output, args.chunk_size)


if __name__ == "__main__":
    main()