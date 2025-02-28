import polars as pl
import pandas as pd 
import argparse

def csv_to_parquet(csv_path, parquet_path):
    df_polars = pl.read_csv(csv_path, separator= ';', try_parse_dates=True, ignore_errors=True, infer_schema_length=1000)
    df_polars.write_parquet(parquet_path, compression="zstd")

def main():
    parser = argparse.ArgumentParser('convert the csv files to parquet')
    parser.add_argument('--csvpath', type = str, required = True, help= 'path for the parquet files')
    parser.add_argument('--parquetpath', type = str, required = True, help= 'path for the parquet files')
    #parser.add_argument('--sensor_columns', type=str, required = False, help="Comma-separated list of columns (e.g., '03091002_x','03091003_x')")
    args = parser.parse_args()
    csv_path = args.csvpath
    parquet_path = args.parquetpath
    #sensor_cols= [str(x) for x in args.sensor_columns.split(',')]
    
    csv_to_parquet(csv_path, parquet_path)



if __name__ == '__main__':
    main()