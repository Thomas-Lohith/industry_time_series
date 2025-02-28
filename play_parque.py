import polars as pl
import matplotlib.pyplot as plt, mpld3
import numpy as np
import pandas as pd


def read_data(path: str, cols: list[str]) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=cols)
    print(df.describe())
    return df

def filter_dc_by_mean(df: pl.DataFrame, sensor_columns: list[str]) -> pl.DataFrame:

    time = df.select('time')
    df_notime = df.drop('time')
    df = df.drop_nulls()
    # Subtract the mean from each column
    df_notime = df_notime.with_columns( (pl.all() - pl.all().mean()))

    df = pl.concat([df_notime, time], how = 'horizontal' )
    print(df.head())
    # Update the original dataframe with the transformed values, polars doesn't modify in place
    return df

def plot_sensors(df):
    time = df['time'][:2000].to_list()
    sensor1 = df['0309101E_x'][84000:86000].to_list()
    sensor2 = df['030910F6_x'][84000:86000].to_list()

    plt.plot(time, sensor1, marker='.')
    plt.plot(time, sensor2, color='r', marker='.')
    plt.xlabel('time')
    plt.ylabel('Acceleration')
    plt.title('time vs acceleration sensor: 0309101E_x')
    plt.show()


def main():

    #to be written aa parameters
    path = '/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.parquet'
    cols =['time', '0309101E_x', '030910F6_x']
    df = read_data(path, cols)
    df_no_dc = filter_dc_by_mean(df, cols)
    plot_sensors(df_no_dc)




if __name__ == '__main__':
    main()
