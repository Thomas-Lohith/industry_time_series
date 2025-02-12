import polars as pl
import pandas as pd 

pl.read_csv('/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.csv').write_parquet('/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.parquet', compression="zstd")

cols = ['time', '03091002_x', '03091003_x']

df = pl.read_parquet('/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.parquet', columns=cols) #, engine='pyarrow')

print(df.describe())
df.height