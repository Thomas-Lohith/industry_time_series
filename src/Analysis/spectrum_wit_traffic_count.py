import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime, timedelta
from spec import get_vehicle_counts

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def parse_time_column(df):
    df["time"] = pd.to_datetime(df["time"], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
    return df


def load_hourly_sensor_file(root_dir, date, hour):
    """
    Locate hourly CSV file inside YYYYMMDD/csv_acc
    """
    date_folder = os.path.join(root_dir, date, "csv_acc")

    for file in os.listdir(date_folder):
        if f"_{hour:02d}-00-00_" in file:
            return os.path.join(date_folder, file)

    raise FileNotFoundError("Hourly CSV not found")


def extract_15min_signal(csv_path, sensor_id, start_time):
    """
    Extract exact 15-minute window
    """
    df = pd.read_csv(csv_path, sep=";")
    df = parse_time_column(df)

    end_time = start_time + timedelta(minutes=15)

    mask = (df["time"] >= start_time) & (df["time"] < end_time)
    signal_data = df.loc[mask, sensor_id].dropna()

    return signal_data, start_time, end_time


def extract_vehicle_counts(excel_path, start_time):
    """
    Match traffic counts for same 15-min interval
    """
    traffic = pd.read_excel(excel_path)
    traffic["Data e ora"] = pd.to_datetime(traffic["Data e ora"], dayfirst=True)

    row = traffic[traffic["Data e ora"] == start_time]

    if row.empty:
        return None

    return row.iloc[0].drop("Data e ora").to_dict()


# ---------------------------------------------------------
# Spectrum plot
# ---------------------------------------------------------

def plot_15min_spectrum(
    x,
    fs,
    vehicle_counts,
    sensor_id,
    start_time,
    end_time,
    freq_limit=20,
    save_path=None
):
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=2048,
        scaling="density"
    )

    mask = f <= freq_limit
    f, Pxx = f[mask], Pxx[mask]
    Pxx *= 1e6  # µ-units for clarity

    plt.figure(figsize=(12, 6))
    plt.semilogy(f, Pxx, linewidth=1.5)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD × 1e6 [V²/Hz]")
    plt.title(
        f"15-min Power Spectrum | {sensor_id}\n"
        f"{start_time} → {end_time}"
    )

    # ---- vehicle label block ----
    label_text = "\n".join([f"{k}: {int(v)}" for k, v in vehicle_counts.items()])
    plt.text(
        0.02, -0.35,
        label_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run_15min_spectrum_pipeline(
    root_folder,
    traffic_excel,
    date,
    hour,
    start_time_str,
    sensor_id,
    output_dir
):
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    
    csv_path = load_hourly_sensor_file(root_folder, date, hour)
    x, t0, t1 = extract_15min_signal(csv_path, sensor_id, start_time)

    time_part = start_time_str.split(" ")[1]
    vehicle_counts = get_vehicle_counts(traffic_excel, date_str= date, start_time_str=time_part)
    #vehicle_counts = extract_vehicle_counts(traffic_excel, start_time)

    if vehicle_counts is None:
        raise ValueError("No matching traffic data found")

    os.makedirs(output_dir, exist_ok=True)

    plot_15min_spectrum(
        x=x,
        fs=100,
        vehicle_counts=vehicle_counts,
        sensor_id=sensor_id,
        start_time=t0,
        end_time=t1,
        save_path=os.path.join(output_dir, f"{date}_{hour:02d}_{sensor_id}.png")
    )


def main():
    run_15min_spectrum_pipeline(
        root_folder="/Users/thomas/Data/Data_sensors",
        traffic_excel="/data/pool/c8x-98x/traffic_data/7_AID_webcam_data/Febbraio/classi.xlsx",
        date="20250208",
        hour=23,
        start_time_str="2025-02-08 23:00:00",
        sensor_id="030911EF_x",
        output_dir="graphs/15min_spectra"
    )


if __name__ == "__main__":
    main()