import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =====================================
# Calculate num. of water droplet variation by eometrical Optics Rainbow Model
# and also including Log-Normal Droplet Size Distribution, Volume-Constrained Droplet Number Model, etc.
# 1. ENVIRONMENT PARAMETERS
# =====================================
V0 = 1e-6              # total water volume [m^3] = 1 mL
T_air = 298            # air temperature [K]
wind_speed = 2.0       # m/s
sun_elevation = np.deg2rad(30)  # realistic sun height

# Time
dt = 1.0
frames = 150

# =====================================
# 2. DROPLET DISTRIBUTION (REALISTIC)
# =====================================
mu = np.log(0.2e-3)     # mean droplet size (log-normal)
sigma = 0.35            # spray variability
num_bins = 60

d = np.logspace(np.log10(0.03e-3), np.log10(0.6e-3), num_bins)

# =====================================
# 3. PHYSICS COEFFICIENTS
# =====================================
k_evap = 5e-10          # slower evaporation (realistic)
k_wind = 0.015          # wind loss
eye_gamma = 0.6         # eye contrast compression

# =====================================
# 4. STORAGE
# =====================================
d_hist = []
I_hist = []

V = V0

# =====================================
# 5. PRECOMPUTE EVOLUTION
# =====================================
for t in range(frames):

    # --- Evaporation (slows for small droplets)
    evap = k_evap * (T_air - 273) * np.sqrt(d)
    d = np.maximum(d - evap * dt, 0.03e-3)

    # --- Wind volume loss (smooth)
    V *= np.exp(-k_wind * wind_speed * dt)

    # --- Droplet number distribution
    pdf = np.exp(-(np.log(d) - mu)**2 / (2 * sigma**2))
    pdf /= pdf.sum()
    N = (6 * V / (np.pi * d**3)) * pdf

    # --- Sun angle visibility
    theta = sun_elevation + 0.05 * np.sin(0.05 * t)
    visibility = np.clip(np.cos(theta - np.deg2rad(42)), 0, 1)

    # --- Perceived intensity (eye-weighted)
    I_eye = N * d**2 * visibility
    I_eye = I_eye**eye_gamma

    d_hist.append(d * 1e3)
    I_hist.append(I_eye)

# =====================================
# 6. ANIMATION
# =====================================
fig, ax = plt.subplots(figsize=(7, 5))

def update(i):
    ax.clear()

    # RGB rainbow mapping
    colors = plt.cm.rainbow(
        (d_hist[i] - d_hist[i].min()) /
        (d_hist[i].max() - d_hist[i].min())
    )

    ax.scatter(
        d_hist[i],
        I_hist[i],
        c=colors,
        s=25,
        alpha=0.85
    )

    ax.set_yscale("log")
    ax.set_xlim(0.03, 0.6)
    ax.set_ylim(1e2, 1e7)

    ax.set_xlabel("Droplet diameter (mm)")
    ax.set_ylabel("Perceived rainbow intensity")
    ax.set_title("Rainbow Perception Simulation\n(Eye-Realistic Model)")

    ax.grid(True, which="both", alpha=0.4)

anim = FuncAnimation(fig, update, frames=frames, interval=80)

# =====================================
# 7. SAVE MP4
# =====================================
writer = FFMpegWriter(fps=12, bitrate=2000)
anim.save("rainbow_eye_realistic.mp4", writer=writer)
plt.close()

print("Eye-realistic rainbow simulation complete.")
print("Saved as: rainbow_eye_realistic.mp4")
