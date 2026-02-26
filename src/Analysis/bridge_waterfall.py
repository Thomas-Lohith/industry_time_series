"""
Bridge Accelerometer – 3D Waterfall Plot
Continuous acceleration signals from multiple sensors vs time and distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from spect

# ── Parameters ────────────────────────────────────────────────────────────────
FS            = 100       # Hz
DURATION      = 10.0      # seconds
N_SENSORS     = 15
SENSOR_SPACING = 8        # metres

# ── Synthetic continuous signals (replace with your real data) ────────────────
rng = np.random.default_rng(42)
t   = np.linspace(0, DURATION, int(DURATION * FS))
n   = len(t)

def make_signal(sensor_idx, rng):
    """get a continuous acceleration signals for the  sensors givenb."""
    return sig

signals = np.array([make_signal(s, rng) for s in range(N_SENSORS)])  # (N_SENSORS, n)
distances = np.arange(N_SENSORS) * SENSOR_SPACING                     # metres

# ── 3D Waterfall Plot ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#06101f")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#06101f")

# Colour map: sensors coloured by distance
cmap   = plt.get_cmap("plasma")
colors = [cmap(i / (N_SENSORS - 1)) for i in range(N_SENSORS)]

# Downsample time axis for speed
step   = 2
t_ds   = t[::step]
sig_ds = signals[:, ::step]

for i in range(N_SENSORS - 1, -1, -1):   # back to front for correct overlap
    d   = distances[i]
    sig = sig_ds[i]
    col = colors[i]

    # filled polygon (waterfall ribbon): close path at baseline
    verts_x = np.concatenate([[t_ds[0]], t_ds, [t_ds[-1]]])
    verts_z = np.concatenate([[0],       sig,  [0]])
    verts_y = np.full_like(verts_x, d)

    verts = [list(zip(verts_x, verts_y, verts_z))]
    poly  = Poly3DCollection(verts, alpha=0.35, zorder=i)
    poly.set_facecolor((*col[:3], 0.25))
    poly.set_edgecolor((*col[:3], 0.0))
    ax.add_collection3d(poly)

    # line on top
    ax.plot(t_ds, [d] * len(t_ds), sig, color=col, linewidth=0.9, alpha=0.95, zorder=i + N_SENSORS)

# ── Axes & labels ─────────────────────────────────────────────────────────────
ax.set_xlabel("Time  (s)",              color="#7090c0", fontsize=10, labelpad=10)
ax.set_ylabel("Distance  (m)",          color="#7090c0", fontsize=10, labelpad=10)
ax.set_zlabel("Acceleration  (m/s²)",   color="#7090c0", fontsize=10, labelpad=8)
ax.set_title("Bridge Accelerometers – 3D Waterfall",
             color="#c0d4f0", fontsize=13, fontweight="bold", pad=18)

ax.set_xlim(t_ds[0], t_ds[-1])
ax.set_ylim(distances[0], distances[-1])

ax.tick_params(colors="#7090c0", labelsize=7.5)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#1a2a45")

ax.xaxis.line.set_color("#1a2a45")
ax.yaxis.line.set_color("#1a2a45")
ax.zaxis.line.set_color("#1a2a45")
ax.grid(True, color="#1a2a45", linewidth=0.4)

# sensor distance ticks
ax.set_yticks(distances[::3])
ax.set_yticklabels([f"{int(d)}m" for d in distances[::3]], fontsize=7, color="#7090c0")

# colourbar (sensor index → distance)
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(distances[0], distances[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=18)
cbar.set_label("Sensor distance (m)", color="#7090c0", fontsize=8)
cbar.ax.yaxis.set_tick_params(colors="#7090c0", labelsize=7)
cbar.outline.set_edgecolor("#1a2a45")

ax.view_init(elev=28, azim=-55)

plt.tight_layout()
plt.savefig("waterfall_3d.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: waterfall_3d.png")
plt.show()