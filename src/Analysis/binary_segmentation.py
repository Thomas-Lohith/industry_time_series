import pandas as pd
import argparse 


def sensor_data(path, sensor_column):
    # Load data from parquet file with specified columns
    col = [sensor_column, 'time']
    if path.endswith('.csv'):
        df = pd.read_csv(path, sep=';')
    if path.endswith('.parquet'):
        df = pd.read_parquet(path, engine='pyarrow', columns=col)
    return df

def filter_dc_by_mean(df, sensor_column):
    print(df.head())
    print(sensor_column)
    signal = df[f'{sensor_column}'].dropna()
    signal = signal - signal.mean()
    df[f'{sensor_column}'] = signal
    return df


def main():
    parser = argparse.ArgumentParser(description="create a spectogram, using sensor data using for specific columns")
    parser.add_argument('--path', type=str, required=True, help="Path to file")
    parser.add_argument("--sensor", type=str, required=True, help="Sensor column name(s) to process")
    args = parser.parse_args()

    path = args.path
    sensor_column = args.sensor

    df = sensor_data(path, sensor_column)


if __name__ == "__main__":
    main()