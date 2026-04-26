import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
import DELCgen as DELC

route = "/home/mand/DELC_LSP"
datfile = "output.txt"

tbin = 7
datalc = DELC.Load_Lightcurve(route + "/" + datfile, tbin)

time = datalc.time
obs_flux = datalc.flux

# -----------------------------
# Frequency grid
# -----------------------------
T = time.max() - time.min()
dt = np.median(np.diff(time))

min_freq = 1.0 / T
max_freq = 1.0 / (2.0 * dt)

# -----------------------------
# Percentile levels
# -----------------------------
levels = np.unique(np.concatenate([
    np.linspace(0, 90, 46),
    np.linspace(90, 99, 91),
    np.linspace(99, 100, 101)
]))

# -----------------------------
# Storage for percentiles
# -----------------------------
all_powers = []

# -----------------------------
# Run simulations
# -----------------------------
N_sims = 1000

for i in range(N_sims):

    sim = datalc.Simulate_DE_Lightcurve()

    ls = LombScargle(time, sim.flux)
    frequency, power = ls.autopower(minimum_frequency=min_freq, 
                                maximum_frequency=max_freq)

    all_powers.append(power)

# convert once at end (still manageable memory)
all_powers = np.array(all_powers)

# compute significance
significance = np.percentile(all_powers, levels, axis=0)

# -----------------------------
# Save ONLY final result
# -----------------------------
np.savez_compressed(
    'lsp_significance_levels.npz',
    frequency=frequency,
    levels=levels,
    data=significance
)

print "Saved significance matrix only."

# -----------------------------
# Load significance file
# -----------------------------
datafile = np.load("lsp_significance_levels.npz")

frequency = datafile["frequency"]
levels = datafile["levels"]
significance = datafile["data"]

# -----------------------------
# Compute real LSP
# -----------------------------
ls = LombScargle(time, obs_flux)
frequency, real_power = ls.autopower(minimum_frequency=min_freq, 
                                maximum_frequency=max_freq)

# -----------------------------
# Select significance levels
# -----------------------------
target_levels = [95, 99, 99.9]

sig_curves = {}
for lvl in target_levels:
    idx = np.argmin(np.abs(levels - lvl))
    sig_curves[lvl] = significance[idx]
    
# -----------------------------
# Find peak
# -----------------------------
peak_idx = np.argmax(real_power)
peak_freq = frequency[peak_idx]
peak_power = real_power[peak_idx]
peak_period = 1.0 / peak_freq

print "Peak frequency:", peak_freq
print "Peak period:", peak_period
print "Peak power:", peak_power

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

# Real LSP
plt.plot(frequency, real_power, color='black', lw=1.5, label='Real Data')

# Significance curves
for lvl in target_levels:
    plt.plot(frequency, sig_curves[lvl], linestyle='--', label=str(lvl) + '%')

# Mark peak
plt.axvline(peak_freq, color='red', linestyle='--',label= 'Peak: {peak_freq:.6f}')

# Annotate peak
label_text = 'Peak\nP = %.3f\nf = %.8f' % (peak_period, peak_freq)

plt.text(
    peak_freq + 0.001,
    plt.ylim()[1] * 0.9,
    'f = %.4f day$^{-1}$\nP = %.1f days' % (peak_freq, peak_period),
    color='blue',
    fontweight=700,
    bbox=dict(facecolor='white', alpha=0.5, edgecolor='red')
)

# Formatting
plt.xlabel('Frequency (day$^{-1}$)')
plt.ylabel("Lomb-Scargle Power")
plt.title("Lomb-Scargle Periodogram with Significance Levels")
plt.grid(True, which='both', linestyle='-', alpha=0.5)
plt.legend(loc='upper right')

#plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('LSP_significance.png', dpi=300)
plt.show()
