import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# 1. Load the data exported from your Python 2 session

data = np.load('simulated_data.npy', allow_pickle=True, encoding='latin1').item()


time = data['time']        # The time array from j0808
all_sims = data['fluxes']  # The list of 500-1000 simulated flux arrays
obs_flux = data['obs_flux'] # Assuming you saved your real flux too

# 2. Define the frequency grid (Matching your previous plot)
# Minimum freq ~ 1/TotalDuration, Maximum freq ~ 0.07
freq_grid = np.linspace(0.001, 0.07, 1000)

# 3. Calculate the Observed LSP
ls_obs = LombScargle(time, obs_flux)
power_obs = ls_obs.power(freq_grid)

# 4. Calculate LSP for every simulation
print("Processing simulations...")
all_powers = []
for s in all_sims:
    ls_sim = LombScargle(time, s)
    p = ls_sim.power(freq_grid)
    all_powers.append(p)

# Convert to a 2D array for easy percentile calculation
# Shape: (N_sims, N_frequencies)
power_stack = np.array(all_powers)

# 5. Calculate Significance Thresholds
sig_95 = np.percentile(power_stack, 95, axis=0)
sig_99 = np.percentile(power_stack, 99, axis=0)
sig_999 = np.percentile(power_stack, 99.9, axis=0)

# 6. Create the Final Plot
plt.figure(figsize=(12, 6))

# Plot the real data
plt.plot(freq_grid, power_obs, color='blue', label='LSP Power (j0808.2-0751)', lw=1.5)

# Plot the significance lines
plt.plot(freq_grid, sig_95, color='gray', linestyle='--', label='95% Red Noise (MC)')
plt.plot(freq_grid, sig_99, color='red', linestyle='--', label='99% Red Noise (MC)')
plt.plot(freq_grid, sig_999, color='black', linestyle=':', label='99.9% Red Noise (MC)')

# Formatting to match your original plot
plt.xlabel('Frequency (day$^{-1}$)')
plt.ylabel('LSP Power')
plt.title('Lomb-Scargle Periodogram with Red Noise Significance')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.075)
plt.ylim(0, max(power_obs)*1.2)
plt.savefig("LSP_Significance.png", dpi=350)
plt.show()

