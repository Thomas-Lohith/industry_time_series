import numpy as np
import matplotlib.pyplot as plt

# Create time axis
t = np.linspace(0, 0.5, 1000)

# Create true signal: zero until 0.25s, then a sine burst
true_data = np.zeros_like(t)
impact_start = t > 0.25
true_data[impact_start] = 3 * np.sin(2 * np.pi * 40 * (t[impact_start] - 0.25))

# Add noise to create noisy signal
noise = np.random.normal(0, 0.3, len(t))
noisy_data = true_data + noise

# Define thresholds
threshold1 = 1
threshold2 = 2

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(t, true_data, label='true data', color='blue')
plt.plot(t, noisy_data, label='noisy data', color='orange', alpha=0.7)
plt.axvline(x=0.25, color='blue', lw=3, label='Impact time', linestyle='--')

# Plot thresholds
plt.axhline(threshold1, color='deepskyblue', linestyle='--', label='Threshold 1')
plt.axhline(-threshold1, color='deepskyblue', linestyle='--')
plt.axhline(threshold2, color='green', linestyle='--', label='Threshold 2')
plt.axhline(-threshold2, color='green', linestyle='--')

# Annotate faulty triggering
plt.annotate('Unnecessary triggering',
             xy=(0.05, noisy_data[int(0.05*len(t))]),
             xytext=(0.01, 2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10)

plt.annotate('Delay of triggering',
             xy=(0.26, noisy_data[int(0.26*len(t))]),
             xytext=(0.28, -2),
             arrowprops=dict(facecolor='brown', arrowstyle='->'),
             fontsize=10)

plt.title('Illustration of faulty triggering in noisy data')
plt.xlabel('time, s')
plt.ylabel('acc, g')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
