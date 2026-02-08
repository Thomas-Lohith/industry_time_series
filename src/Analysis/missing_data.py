import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

TIME_BIN = "1min"


# =========================
# CORE ANALYSIS
# =========================
def analyze_missing_data(root_dir, sensor_name):
    """
    Efficiently scans large directory trees and computes:
    - % missing data per day
    - day x time-of-day missingness heatmap
    """

    daily_missing_pct = {}
    daily_time_missing = {}

    day_folders = [
        d for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for day in tqdm(day_folders, desc="Scanning days", unit="day"):
        day_path = os.path.join(root_dir, day)
        csv_acc_path = os.path.join(day_path, "csv_acc")

        if not os.path.exists(csv_acc_path):
            continue

        total_samples = 0
        total_missing = 0

        # store minute-level missing fractions
        minute_accumulator = {}

        files = [
            f for f in sorted(os.listdir(csv_acc_path))
            if f.endswith(".csv")
        ]

        for file in tqdm(files, desc=f"{day}", leave=False, unit="file"):
            file_path = os.path.join(csv_acc_path, file)

            try:
                df = pd.read_csv(
                    file_path,
                    sep=";",
                    usecols=["time", sensor_name],
                    dtype={sensor_name: "float32"},
                )
            except Exception:
                continue

            if df.empty:
                continue

            df["time"] = pd.to_datetime(df["time"], format='%Y/%m/%d %H:%M:%S:%f', errors="coerce", exact=False)
            df = df.dropna(subset=["time"])

            if df.empty:
                continue

            missing_mask = df[sensor_name].isna()

            total_samples += len(df)
            total_missing += missing_mask.sum()

            # ---- minute binning ----
            time_bins = df["time"].dt.floor(TIME_BIN)

            for t, m in zip(time_bins, missing_mask):
                key = t.time()
                if key not in minute_accumulator:
                    minute_accumulator[key] = [0, 0]
                minute_accumulator[key][0] += int(m)
                minute_accumulator[key][1] += 1

        if total_samples == 0:
            continue

        # % missing per day
        daily_missing_pct[day] = 100 * total_missing / total_samples

        # minute-level missing fraction
        daily_time_missing[day] = {
            t: miss / cnt
            for t, (miss, cnt) in minute_accumulator.items()
        }

    return daily_missing_pct, daily_time_missing


# =========================
# VISUALIZATION
# =========================
def plot_missing_percentage(daily_missing_pct):
    plt.figure(figsize=(12, 4))
    plt.bar(daily_missing_pct.keys(), daily_missing_pct.values())
    plt.xticks(rotation=90)
    plt.ylabel("% Missing")
    plt.title("Percentage of Missing Data per Day")
    plt.tight_layout()
    plt.show()


def plot_missing_heatmap(daily_time_missing):
    heatmap_df = pd.DataFrame(daily_time_missing).T.sort_index()

    plt.figure(figsize=(14, 6))
    plt.imshow(heatmap_df, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Fraction Missing")
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.xlabel("Time of Day")
    plt.ylabel("Date")
    plt.title("Missing Data Heatmap (Day × Time)")
    plt.tight_layout()
    plt.show()


def plot_missing_probability(daily_time_missing):
    heatmap_df = pd.DataFrame(daily_time_missing)
    missing_prob = heatmap_df.mean(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(missing_prob.index.astype(str), missing_prob.values)
    plt.xticks(rotation=45)
    plt.ylabel("Missing Probability")
    plt.xlabel("Time of Day")
    plt.title("Missing Probability vs Time of Day")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    ROOT_DIR = "/Users/thomas/Data/Data_sensors"
    SENSOR_NAME = "03091002_x"

    daily_missing_pct, daily_time_missing = analyze_missing_data(
        ROOT_DIR, SENSOR_NAME
    )

    plot_missing_percentage(daily_missing_pct)
    plot_missing_heatmap(daily_time_missing)
    #plot_missing_probability(daily_time_missing)