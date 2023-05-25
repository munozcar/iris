# IRIS: InfraRed Isothermal Slabs

IRIS is a Python package that simulates IR and mid-IR molecular line emission from protoplanetary disks. 

<img align="left" src="https://github.com/munozcar/IRIS/assets/32044135/c045b724-755e-4f9e-b086-308ff66c098d"  width="150" height="190">

The emission lines are modeled using isothermal slabs, with a detailed wavelength-dependent opacity
treatment that accounts for overlapping lines and saturation effects. 

IRIS is vectorized and optimized with jax, i.e. the code is dynamically compiled and can be significantly sped-up using GPUs.

One can use either a single slab, or a set of slabs to reproduce the effects of radial temperature and density gradients in the disk.

Basic package requirements: numpy, scipy, astropy, jax, pandas. Additional package requirements: jaxns, tinyGP, jaxopt. These are needed to do inference with IRIS. 

You need at least 16GB of RAM, and ideally (but not necessarily) a GPU. You need to unzip HITRAN.zip before use.

#### Examples:
To model emission from CO and H2O using one slab for each:
```
import iris as iris
import numpy as np

# fine wavelength grid (in micron) to evaluate opacities
fine_wgrid = np.arange(4.7,8.6,6e-5)
# wavelength grid (in micron) to downsample
obs_wgrid = np.arange(4.8,8.5,0.002)
# resolving power 
R = 3200

# DISK MODEL

distance = 120 # pc

# excitation temperatures in K
T_ex =  np.array([np.array([600.0]), # for CO
                  np.array([800.0])]) # for H2O
# column densities in cm^-2
N_mol = np.array([np.array([1e17]), # for CO
                  np.array([1e16])]) # for H2O
# emitting areas in au^2
A_au =  np.array([np.array([2.0]), # for CO
                  np.array([1.0])]) # for H2O
# intrinsic line widths in km/s
dV =    np.array([np.array([2.0]), # for CO
                  np.array([2.0])]) # for H2O

# model spectrum
slab = iris.slab(molecules=['CO', 'H2O'], wlow=4.8, whigh=26.0)
slab.setup_disk(distance, T_ex, N_mol, A_au, dV)
slab.setup_grid(fine_wgrid, obs_wgrid, R)
slab.simulate()

# Plot slab.downsampled_flux:
```

<img src="https://github.com/munozcar/IRIS/assets/32044135/b1b92e02-1c82-4144-8398-b557075c2c02"  width="600" height="200">


