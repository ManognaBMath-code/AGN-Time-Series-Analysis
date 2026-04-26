import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import DELCgen as DELC
from astropy.stats import LombScargle
"""
# --------------------------------------------------
# 1. USER PARAMETERS
# --------------------------------------------------
route = "/home/mand/DELC_WWZ"
datfile = "output.txt"
tbin = 7

N_sims = 1000
wwz_script = "wwz.py"

# decay constant (same for all runs)
DCON = 0.001

# frequency resolution (for WWZ grid step)
freq_step = 0.0001

# --------------------------------------------------
# 2. LOAD LIGHT CURVE
# --------------------------------------------------
datalc = DELC.Load_Lightcurve(route + "/" + datfile, tbin)

time = datalc.time
flux = datalc.flux

# --------------------------------------------------
# 3. GET LSP-BASED FREQUENCY RANGE 
# --------------------------------------------------
T = time.max() - time.min()
dt = np.median(np.diff(time))

freq_low = 1.0 / T
freq_high = 1.0 / (2.0 * dt)

print "Frequency range from data:"
print "low:", freq_low
print "high:", freq_high

# --------------------------------------------------
# 4. REAL WWZ RUN (via CLI)
# --------------------------------------------------
real_outfile = "wwz_real.dat"

cmd = [
    "python", wwz_script,
    "-f", route + "/" + datfile,
    "-o", real_outfile,
    "-l", str(freq_low),
    "-hi", str(freq_high),
    "-d", str(freq_step),
    "-c", str(DCON),
    "-p", "0"
]

subprocess.call(cmd)

print "Real WWZ done"

# --------------------------------------------------
# 5. SIMULATIONS
# --------------------------------------------------
sim_files = []
wwz_files = []

for i in range(N_sims):

    sim_file = "sim_%d.txt" % i
    wwz_file = "wwz_%d.dat" % i
    sim_lc = datalc.Simulate_DE_Lightcurve()
    
    np.savetxt(sim_file, np.column_stack((time, sim_lc.flux)))

    cmd = [
        "python", wwz_script,
        "-f", sim_file,
        "-o", wwz_file,
        "-l", str(freq_low),
        "-hi", str(freq_high),
        "-d", str(freq_step),
        "-c", str(DCON),
        "-p", "0"
    ]

    subprocess.call(cmd)

sim_files.append(sim_file)
wwz_files.append(wwz_file)

if i % 100 == 0:
    print "Simulation:", i
"""
# --------------------------------------------------
# 6. LOAD WWZ OUTPUTS
# --------------------------------------------------
def load_wwz(file):
    data = np.loadtxt(file)
    # expected format: time, freq, power
    return data

real = load_wwz("wwz_real.dat")
tau = np.unique(real[:, 0])
freq = np.unique(real[:, 1])
wwz_raw = real[:, 2]

wwz_real = wwz_raw.reshape(len(tau), len(freq)).T


import glob
import re

def extract_num(f):
    return int(re.findall(r'\d+', f)[0])

wwz_files = sorted(
    glob.glob("wwz_[0-9]*.dat"),  # excludes wwz_real.dat
    key=extract_num
)

print "Found WWZ files:", len(wwz_files)
all_wwz = []

for f in wwz_files:
    data = load_wwz(f)
    if data.size == 0:
        print "Skipping empty file:", f
        continue
    if data.ndim != 2 or data.shape[1] < 3:
        print "Skipping invalid file:", f
        continue
    data = data[np.lexsort((data[:,1], data[:,0]))]
    tau_sim = np.unique(data[:, 0])
    freq_sim = np.unique(data[:, 1])
    wwz_sim = data[:, 2]
    wwz_sim_grid = wwz_sim.reshape(len(tau_sim), len(freq_sim)).T
    proj = np.mean(wwz_sim_grid, axis=1)
    all_wwz.append(proj)

all_wwz = np.array(all_wwz)

print "Shape:", all_wwz.shape
# --------------------------------------------------
# 7. SIGNIFICANCE (PERCENTILES)
# --------------------------------------------------

# WWZ frequency projection (real data)
wwz_freq = np.mean(wwz_real, axis=1)

levels = np.unique(np.concatenate([
    np.linspace(0, 90, 46),
    np.linspace(90, 99, 91),
    np.linspace(99, 100, 101)
]))

significance = np.percentile(all_wwz, levels, axis=0)

np.savez_compressed("wwz_significance.npz",tau=tau,frequency=freq,levels=levels,data=significance)

print "Saved significance file"


# --------------------------------------------------
# 8. FREQUENCY PROJECTION
# --------------------------------------------------

data = np.load("wwz_significance.npz")

tau = data["tau"]
freq = data["frequency"]
levels = data["levels"]
significance = data["data"]

target_levels = [95, 99, 99.9]

# store full frequency-projected significance curves
sig_freq = {}

for lvl in target_levels:
    idx = np.argmin(np.abs(levels - lvl))
    # try for this time
    if significance.ndim == 3:
        sig_freq[lvl] = np.mean(significance[idx], axis=1)
    elif significance.ndim == 2:
        sig_freq[lvl] = significance[idx]
    else:
        raise ValueError("Unexpected significance shape")


# --------------------------------------------------
# 9. PEAK DETECTION
# --------------------------------------------------

peak_idx = np.argmax(wwz_freq)
peak_freq = freq[peak_idx]
peak_period = 1.0 / peak_freq

print "Peak frequency:", peak_freq
print "Peak period:", peak_period


# --------------------------------------------------
# 10. PLOT
# --------------------------------------------------
plt.figure(figsize=(14, 5))

# WWZ map
plt.subplot(1, 2, 1)

if wwz_real.shape == (len(tau), len(freq)):
    wwz_real = wwz_real.T
    
plt.contourf(tau, freq, wwz_real, 100, cmap='viridis')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("WWZ_Map")

# WWZ Power spectrum
plt.subplot(1, 2, 2)

plt.plot(wwz_freq, freq, 'b', label='Average WWZ')
plt.plot(sig_freq[95], freq, 'b--', label='95% C.L.')
plt.plot(sig_freq[99], freq, 'g--', label='99% C.L.')
plt.plot(sig_freq[99.9], freq, 'r--', label='99.9% C.L.')
plt.axhline(peak_freq, color='blue', linestyle=':')

#plt.text(np.max(wwz_freq) + 0.001,peak_freq,'Peak\nf=%.4f\nP=%.2f' % (peak_freq, peak_period),color='blue', bbox=dict(facecolor='white', alpha=0.5, edgecolor='red')
plt.text(
    0.5 0.95,   # position (right-top corner)
    'Peak\nf=%.4f\nP=%.2f' % (peak_freq, peak_period),
    transform=plt.gca().transAxes,  
    ha='right',  
    va='top',
    fontsize=10,
    color='blue',
    bbox=dict(
        facecolor='white',
        edgecolor='black',
        boxstyle='round,pad=0.3'
    )
)
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("WWZ Frequency Projection")
plt.legend()

plt.tight_layout()
plt.savefig('WWZ_Significance.png', dpi=300)
plt.show()

"""
#---------------------------------------------------------
# 11. Remove Temporary Files
#---------------------------------------------------------
import glob
import os

# remove all simulation files
for f in glob.glob("sim_*.txt"):
    os.remove(f)

# remove all WWZ files
for f in glob.glob("wwz_*.dat"):
    os.remove(f)

print "All temporary files deleted"
"""
