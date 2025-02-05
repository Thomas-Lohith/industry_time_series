import os
import pandas as pd
import argparse

def merge_csv_files(directory, output_file="one_day.csv"):
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return
    
    # Get all CSV files from the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('_th.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    one_hr_dataframes = []
    
    # Read and concatenate all CSV files
    for file in csv_files:
        file_path = os.path.join(directory, file)
        print(f"Processing file: {file_path}")

        try:
            df = pd.read_csv(file_path, sep=';')  # Removed `sep=';'` for flexibility
            df_col = df[['time', '0309101E_x']]  # Keep only required columns
            
            one_hr_dataframes.append(df_col)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Merge all dataframes
    oneday_df = pd.concat(one_hr_dataframes, ignore_index=True)

    # Convert time column to datetime
    oneday_df['time'] = pd.to_datetime(oneday_df['time'], format='%Y/%m/%d %H:%M:%S:%f', errors='coerce')

    # Save to CSV
    oneday_df.to_csv(output_file, index=False)
    print(f"Successfully merged CSV files into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge all CSV files in a given directory.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the folder containing CSV files.")
    args = parser.parse_args()

    merge_csv_files(args.file_path)

if __name__ == "__main__":
    main()