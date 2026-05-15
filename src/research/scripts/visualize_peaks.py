
"""
visualize_peaks.py
------------------
Reads res.csv and generates one PNG per vehicle showing its
traversal across bridge sensors.

  X  = sensor ID (ordered by first activation = bridge traversal order)
  Y  = seconds since vehicle first peak  (time flows downward)
  Colour = amplitude  (diverging RdBu: red=positive, blue=negative)

Usage:
    python visualize_peaks.py                          # uses res.csv by default
    python visualize_peaks.py --input my_file.csv --output_dir output/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
import json


# ── 1. Parse peaks ────────────────────────────────────────────────────────────

def parse_peaks_cell(cell):
    """Parse 'ts|amp;ts|amp;...' string into list of (timestamp, float)."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    peaks = []
    for entry in str(cell).split(";"):
        parts = entry.strip().split("|")
        if len(parts) == 2:
            try:
                peaks.append((pd.to_datetime(parts[0].strip()), float(parts[1].strip())))
            except Exception:
                continue
    return peaks


# ── 2. Build long-form dataframe ──────────────────────────────────────────────

def build_long_df(df, sensor_cols, sensor_ids):
    """Explode every peak cell -> one row per (vehicle, sensor, peak)."""
    records = []
    for _, row in df.iterrows():
        vid = row["vehicle_id"]
        for scol, sid in zip(sensor_cols, sensor_ids):
            for ts, amp in parse_peaks_cell(row.get(scol)):
                records.append({
                    "vehicle_id": vid,
                    "sensor_id":  sid,
                    "timestamp":  ts,
                    "amplitude":  amp,
                })

    long = pd.DataFrame(records)
    if long.empty:
        raise ValueError("No peaks found — check that columns end with _dominant_peaks")

    long["timestamp"] = pd.to_datetime(long["timestamp"])

    # Relative seconds per vehicle (t=0 = vehicle's earliest detected peak)
    for vid in long["vehicle_id"].unique():
        mask = long["vehicle_id"] == vid
        t0 = long.loc[mask, "timestamp"].min()
        long.loc[mask, "rel_seconds"] = (
            long.loc[mask, "timestamp"] - t0
        ).dt.total_seconds()

    return long


# ── 3. Sensor ordering ────────────────────────────────────────────────────────

def sensor_activation_order(long_df):
    """Return sensors ordered by earliest peak across ALL vehicles."""
    first = long_df.groupby("sensor_id")["rel_seconds"].min().sort_values()
    return first.index.tolist()


# ── 4. Plot one vehicle ───────────────────────────────────────────────────────

def plot_vehicle(sub, sensor_order, amp_max, vehicle_id, output_path):
    sensor_y = {sid: i for i, sid in enumerate(sensor_order)}
    sub = sub.copy()
    sub["x_pos"] = sub["sensor_id"].map(sensor_y)

    fig, ax = plt.subplots(figsize=(20, 12))

    # Diverging colour map centred at 0
    norm   = mcolors.TwoSlopeNorm(vmin=-amp_max, vcenter=0, vmax=amp_max)
    cmap   = cm.RdBu_r
    colors = cmap(norm(sub["amplitude"].values))

    sc = ax.scatter(
        sub["x_pos"].values,
        sub["rel_seconds"].values,
        c=abs(sub["amplitude"]).values,
        cmap=cmap,
        norm=norm,
        s=60,
        alpha=0.88,
        edgecolors="rgba(30,30,30,0.3)" if False else (0.12, 0.12, 0.12, 0.30),
        linewidths=0.6,
        zorder=3,
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("Amplitude", fontsize=11)
    cbar.ax.tick_params(labelsize=8)

    # Alternating bands every 10 sensors
    for i in range(0, len(sensor_order), 10):
        ax.axvspan(i - 0.5, min(i + 4.5, len(sensor_order) - 0.5),
                   color="grey", alpha=0.04, zorder=0)

    # X axis — sensor labels
    ax.set_xticks(range(len(sensor_order)))
    ax.set_xticklabels(sensor_order, rotation=90, fontsize=7, family="monospace")
    ax.set_xlim(-0.7, len(sensor_order) - 0.3)
    ax.set_xlabel("Sensor ID  (activation order →)", fontsize=12, labelpad=8)

    # Y axis — time, reversed (0 at top)
    ax.set_ylabel("Seconds since first peak", fontsize=12, labelpad=8)
    #ax.invert_yaxis()
    ax.yaxis.set_tick_params(labelsize=9)

    # Grid
    ax.grid(axis="both", color="#cccccc", linewidth=0.4, linestyle="--", zorder=1)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_title(
        f"Vehicle {vehicle_id} — Traversal Across Sensors\n"
        "X = sensor (bridge order)  |  Y = time ↓  |  Colour = amplitude (red +, blue −)",
        fontsize=14, pad=14,
    )

    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {out}")


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One traversal PNG per vehicle from res.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="res.csv",
        help="Input CSV (default: res.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for PNGs (default: output/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Peak Traversal Visualizer")
    print("=" * 60)
    print(f"Input  : {args.input}")
    print(f"Output : {args.output_dir}/")

    # Load
    df = pd.read_csv(args.input)
    print(f"Loaded : {len(df)} vehicle rows")

    # Detect sensor columns
    sensor_cols = [c for c in df.columns if c.endswith("_dominant_peaks")]
    sensor_ids  = [c.replace("_dominant_peaks", "") for c in sensor_cols]
    if not sensor_cols:
        raise ValueError("No columns ending with _dominant_peaks found in CSV.")
    print(f"Sensors: {len(sensor_ids)}")

    # Build long df
    long_df = build_long_df(df, sensor_cols, sensor_ids)
    print(f"Peaks  : {len(long_df)} total")

    # Shared reference values — same scale across all vehicle plots
    sensor_order = sensor_activation_order(long_df)
    amp_max      = long_df["amplitude"].abs().max()

    # One plot per vehicle
    vehicles = sorted(long_df["vehicle_id"].unique())
    print(f"Vehicles: {vehicles}\n")

    for veh in vehicles:
        sub      = long_df[long_df["vehicle_id"] == veh]
        out_path = Path(args.output_dir) / f"{veh}_traversal.png"
        plot_vehicle(sub, sensor_order, amp_max, veh, out_path)

    print("\n" + "=" * 60)
    print(f"Done!  {len(vehicles)} plot(s) saved to  {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()