import polars as pl
import pandas as pd 
import time
#import hvplot.polars # Using the hvplot as the plotting backend
from polars.testing import assert_frame_equal
parquet_path = '/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.parquet'
csv_path = '/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.csv'

# df_polars = pl.read_csv(csv_path, separator= ';', try_parse_dates=True, ignore_errors=True, infer_schema_length=1000)
# df_polars.write_parquet(parquet_path, compression="zstd")

gpu_engine = pl.GPUEngine(
    device=0,  # This is the default
    raise_on_fail=True,  # Fail loudly if can't execute on the GPU
)

cols =['time','03091002_x','03091003_x']

df = pl.read_parquet(parquet_path, columns=cols)

print(df.describe())

print(df.select(pl.col('03091002_x').sum()).collect())