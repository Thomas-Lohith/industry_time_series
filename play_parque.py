import polars as pl
import matplotlib.pyplot as plt, mpld3
import numpy as np
import pandas as pd
import hvplot.polars 

def read_data(path: str, cols: list[str]) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=cols)
    print(df.describe())
    print(df.head())
    return df

# def filter_dc_by_mean(df: pl.DataFrame, sensor_columns: list[str]) -> pl.DataFrame:
#     time = df.select('time')
#     df_notime = df.drop('time')
#     df = df.drop_nulls()
#     #Subtract the mean from each column
#     df_notime = df_notime.with_columns( (pl.all() - pl.all().mean()))

#     df = pl.concat([df_notime, time], how = 'horizontal' )
#     print(df.describe())
#     # Update the original dataframe with the transformed values, polars doesn't modify in place
#     return df

def filter_dc_by_mean(df: pl.DataFrame, sensor_columns: list[str]) -> pl.DataFrame:
    """
    Remove DC offset from specified sensor columns by subtracting the mean of each column.
    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame
    sensor_columns : list[str]
        List of column names containing sensor data to process
    Returns:
    --------
    pl.DataFrame
        DataFrame with DC offset removed from specified sensor columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.clone()
    
    # Calculate mean for each sensor column and subtract it
    for col in sensor_columns:
        if col in df.columns and col != 'time':
            # Only process columns that exist in the DataFrame
            col_mean = df[col].mean()
            result_df = result_df.with_columns(
                (pl.col(col) - col_mean).alias(col)
            )
    
    #print(result_df.describe())
    print(result_df.head())
    return result_df

def plot_sensors(daraframe):
    # time = df['time']
    # sensor1 = df['0309101E_x']

    # plt.plot(time, sensor1, marker='.')
    # plt.plot(time, sensor2, color='r', marker='.')
    # plt.xlabel('time')
    # plt.ylabel('Acceleration')
    # plt.title('time vs acceleration sensor: 0309101E_x')
    # plt.show()

    plot = df.hvplot.line(x = df['time'], y= df['0309101E_x'])
    plot.show()

def main():

    #to be written aa parameters
    path = '/Users/thomas/Desktop/phd_unipv/Industrial_PhD/Data/20241126/csv_acc/combined.parquet'
    cols =['time', '0309101E_x', '030910F6_x']
    df = read_data(path, cols)
    df_no_dc = filter_dc_by_mean(df, cols)
    plot_sensors(df_no_dc)




if __name__ == '__main__':
    main()
