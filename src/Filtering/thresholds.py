import pandas as pd
import polars as pl
import argparse
import numpy as np

def compute_thresholds(input_csv, output_csv, x):
    # Load data
    df = pd.read_csv(input_csv, sep=';')
    
    # Remove timestamp if present
    if "time" in df.columns:
        sensor_cols = [col for col in df.columns if col != "time"]
    else:
        sensor_cols = df.columns.tolist()
    
    results = []
    
    for sensor in sensor_cols:
        mean_val = round(np.abs(df[sensor]).mean(), 6)
        std_val = round(np.abs(df[sensor]).std(), 6)
        
        row = {
            "sensor": sensor,
           # "mean": mean_val,
           # "std": std_val,
        }
        
        #for x in x_values:
        row[f"threshold_x = {x}"] = mean_val + round((float(x) * float(std_val)), 6)
        
        results.append(row)
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Thresholds saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="compute the thresholds, using sensor data for all columns")
    parser.add_argument('--input_csv', type=str, required=True, help="Input csv file")
    parser.add_argument("--output_csv", type=str, required=True, help="Output csv file with the thresholds")
    parser.add_argument('--x', type= int, default=3, help='how many times the standard deviation + mean is the threshold')
    args = parser.parse_args()

    input_csv = args.input_csv
    sensor_thresholds = args.output_csv
    x = args.x

    compute_thresholds(input_csv, sensor_thresholds, x)

if __name__ == "__main__":
    main()


# Example usage:
# compute_thresholds("bridge_data.csv", "sensor_thresholds.csv")
#python3 src/Filtering/thresholds.py --input_csv /Users/thomas/Data/Data_sensors/20241126/csv_acc/M001_2024-11-26_16-00-00_gg-9_int-17_th.csv --output_csv /Users/thomas/Desktop/github_repos/industry_time_series/src/shared/old_thresholds_abs.csv

