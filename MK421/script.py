import matplotlib.pyplot as plt
import matplotlib


from fermipy.gtanalysis import GTAnalysis

gta = GTAnalysis('config.yaml',logging={'verbosity' : 3})
gta.setup()
gta.optimize()
# Free Normalization of all Sources within 3 deg of ROI center
gta.free_sources(distance=3.0,pars='norm')

# Free all parameters of isotropic and galactic diffuse components
gta.free_source('galdiff')
gta.free_source('isodiff')
# Free sources with TS > 10
gta.free_sources(minmax_ts=[10,None],pars='norm')
gta.free_source('mkn421')
gta.fit()
gta.write_roi('fit_model')
gta.sed('mkn421', make_plots=True)
lc = gta.lightcurve('mkn 421', free_radius=3.0, nbins=8, multithread=True, nthread=8, use_scaled_srcmap=True)
"""
print(lc['tmin'])
print(lc['tmax'])
print(lc['fit_success'])
print(lc['ts_var'])
print(lc['flux'])
print(lc['eflux'])
print(lc['flux_ul95'])
plt.clf()

fig = plt.figure(figsize=(8,6))
plt.errorbar((lc['tmin']+lc['tmax'])/2., lc['flux'], yerr=lc['flux_err'], xerr=(lc['tmax']-lc['tmin'])/2., fmt="o", color="black")
plt.ylabel(r'$\Phi_{\gamma}$ [ph/cm$^2$/s]')
plt.xlabel(r'$t$ [s]')

plt.grid(True)
plt.yscale('log')
plt.xscale('linear')
fig.tight_layout(pad=0.5)
plt.savefig("lc.eps")
plt.show()
"""

