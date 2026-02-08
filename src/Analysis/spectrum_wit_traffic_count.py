import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime, timedelta
import os
import argparse


# =========================================================
# FILE DISCOVERY
# =========================================================
def find_csv_file(root_folder, date_str, hour):
    hour1 = hour + 1
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    csv_folder = os.path.join(root_folder, date_str, "csv_acc")

    for fname in os.listdir(csv_folder):
        if (
            fname.startswith(f"M001_{formatted_date}_{hour:02d}-00-00_")
            and f"_int-{hour1}_" in fname
            and fname.endswith("_th.csv")
        ):
            return os.path.join(csv_folder, fname)

    raise FileNotFoundError("No matching CSV file found.")


# =========================================================
# DATA EXTRACTION
# =========================================================
def extract_15min_data(csv_path, sensor_id, start_time_str):
    df = pd.read_csv(csv_path, sep=';')
    df['time'] = pd.to_datetime(
        df['time'],
        format='%Y/%m/%d %H:%M:%S:%f',
        errors="coerce",
        exact=False
    )

    h, m, s = map(int, start_time_str.split(':'))
    base_date = df['time'].iloc[0].date()

    start_dt = datetime.combine(base_date, datetime.min.time().replace(hour=h, minute=m, second=s))
    end_dt = start_dt + timedelta(minutes=15)

    seg = df[(df['time'] >= start_dt) & (df['time'] < end_dt)]

    if seg.empty:
        raise ValueError("No data in interval")

    return seg[sensor_id].values, start_dt, end_dt


# =========================================================
# VEHICLE COUNTS
# =========================================================
def get_vehicle_counts(excel_path, date_str, start_time_str):
    df = pd.read_excel(excel_path) if excel_path.endswith(('.xls', '.xlsx')) else pd.read_csv(excel_path)

    h, m, _ = map(int, start_time_str.split(':'))
    em, eh = (m + 15) % 60, h + (m + 15) // 60
    tr = f"{h}:{m:02d} - {eh}:{em:02d}"

    for _, r in df.iterrows():
        if isinstance(r['Data e ora'], str) and tr in r['Data e ora']:
            return {k: int(r.get(k, 0)) for k in ["Car", "Bus", "Motorbike", "Truck", "Van"]}

    return dict.fromkeys(["Car", "Bus", "Motorbike", "Truck", "Van"], 0)


# =========================================================
# POWER SPECTRUM
# =========================================================
def compute_power_spectrum(data, fs):
    data -= np.mean(data)
    f, p = signal.welch(data, fs=fs, window="hann", nperseg=2048, scaling="density")
    m = f <= 20
    return f[m], p[m] * 1e6


# =========================================================
# 🔹 ADDED: SPECTRAL FEATURES
# =========================================================
def extract_spectral_features(freqs, power):
    return {
        "centroid": np.sum(freqs * power) / np.sum(power),
        "low_0_3Hz": np.trapz(power[(freqs < 3)], freqs[(freqs < 3)]),
        "mid_3_8Hz": np.trapz(power[(freqs >= 3) & (freqs < 8)], freqs[(freqs >= 3) & (freqs < 8)]),
        "high_8_20Hz": np.trapz(power[(freqs >= 8)], freqs[(freqs >= 8)]),
        "total_energy": np.trapz(power, freqs)
    }


# =========================================================
# 🔹 ADDED: SPECTRAL COMPARISON
# =========================================================
def compare_spectra(p1, p2):
    return {
        "mae": np.mean(np.abs(p1 - p2)),
        "pearson_corr": np.corrcoef(p1, p2)[0, 1],
        "cosine_sim": np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    }


# =========================================================
# 🔹 ADDED: STATISTICAL REPORT
# =========================================================
def generate_statistical_report(data_list, out_dir, tag):
    rows = []

    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            f, p1, v1, t1s, t1e = data_list[i]
            _, p2, v2, t2s, t2e = data_list[j]

            r = {
                "seg1": f"{t1s:%H:%M}-{t1e:%H:%M}",
                "seg2": f"{t2s:%H:%M}-{t2e:%H:%M}",
                "veh_diff": abs(sum(v2.values()) - sum(v1.values()))
            }

            r.update(compare_spectra(p1, p2))

            f1, f2 = extract_spectral_features(f, p1), extract_spectral_features(f, p2)
            r["centroid_diff"] = abs(f2["centroid"] - f1["centroid"])
            r["high_band_diff"] = abs(f2["high_8_20Hz"] - f1["high_8_20Hz"])

            rows.append(r)

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"spectral_stats_{tag}.csv")
    df.to_csv(path, index=False)

    print("\n📊 SPECTRAL STATISTICS")
    print(df.round(4))
    print(f"Saved: {path}")

    return df


# =========================================================
# 🔹 ADDED: METRIC PLOTS
# =========================================================
def plot_metric_vs_vehicle_diff(df, out_dir, tag):
    for m in ["mae", "pearson_corr", "cosine_sim", "centroid_diff"]:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["veh_diff"], df[m])
        plt.xlabel("Vehicle count difference")
        plt.ylabel(m.upper())
        plt.grid(True)
        p = os.path.join(out_dir, f"{m}_vs_vehicle_diff_{tag}.png")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")


# =========================================================
# MAIN
# =========================================================
def main(args):
    FS = 100
    OUT = "/home/c8x-98x/industry_time_series/src/results/filtering"
    os.makedirs(OUT, exist_ok=True)

    data_list = []

    for t in args.start_time:
        csv = find_csv_file(args.root_folder, args.date, args.hour)
        x, ts, te = extract_15min_data(csv, args.sensor_id, t)
        vc = get_vehicle_counts(args.excel_path, args.date, t)
        f, p = compute_power_spectrum(x, FS)
        data_list.append((f, p, vc, ts, te))

    # ===== STATISTICS (OPTIONAL) =====
    if args.stats and args.superimpose and len(data_list) > 1:
        tag = f"{args.date}_{args.hour:02d}_{args.sensor_id}"
        df = generate_statistical_report(data_list, OUT, tag)

        if args.stats_plots:
            plot_metric_vs_vehicle_diff(df, OUT, tag)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root-folder", required=True)
    parser.add_argument("-e", "--excel-path", required=True)
    parser.add_argument("-d", "--date", required=True)
    parser.add_argument("-H", "--hour", type=int, required=True)
    parser.add_argument("-s", "--sensor-id", required=True)
    parser.add_argument("-t", "--start-time", action="append", required=True)

    parser.add_argument("--subplot", action="store_true")
    parser.add_argument("--superimpose", action="store_true")

    # 🔹 NEW FLAGS
    parser.add_argument("--stats", action="store_true", help="Generate spectral statistics")
    parser.add_argument("--stats-plots", action="store_true", help="Generate metric plots")

    args = parser.parse_args()
    main(args)