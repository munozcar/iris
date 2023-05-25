# IRIS
IRIS: InfraRed Isothermal Slabs

<img src="https://github.com/munozcar/IRIS/assets/32044135/c045b724-755e-4f9e-b086-308ff66c098d"  width="150" height="190">

Example:
```
# fine wavelength grid (in micron) to evaluate opacities
fine_wgrid = np.arange(4.7,8.6,6e-5)
# wavelength grid (in micron) to downsample
obs_wgrid = np.arange(4.8,8.5,0.002)
# resolving power 
R = 3200

# DISK MODEL
# excitation temperatures in K
T_ex =  np.array([np.array([600.0]),
                  np.array([800.0])])
# column densities in cm^-2
N_mol = np.array([np.array([1e17]),
                  np.array([1e16])])
# emitting areas in au^2
A_au =  np.array([np.array([2.0]),
                  np.array([1.0])])
# intrinsic line widths in km/s
dV =    np.array([np.array([2.0]),
                  np.array([2.0])])

# model spectrum
slab = iris.slab(molecules=['CO', 'H2O'], wlow=4.8, whigh=26.0)
slab.setup_disk(distance, T_ex, N_mol, A_au, dV)
slab.setup_grid(fine_wgrid, obs_wgrid, R)
slab.simulate()
```
