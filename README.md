# iris: InfraRed Isothermal Slabs

iris is a Python package that simulates IR and mid-IR molecular line emission from protoplanetary disks. 

The emission lines are modeled using isothermal slabs, with a detailed wavelength-dependent opacity
treatment that accounts for overlapping lines and saturation effects. 

iris is vectorized and optimized with jax, i.e. the code is dynamically compiled and can (but does not need to) be significantly sped-up using GPUs.

One can use either a single slab, or a set of slabs to reproduce the effects of radial temperature and density gradients in the disk.

Basic package requirements: numpy, scipy, astropy, jax, pandas. 

#### Examples:
To model emission from CO and H2O using a single slab for each:

```
import iris as iris
import numpy as np

# Define a fine wavelength grid (in micron) to evaluate opacities
fine_wgrid = np.arange(4.7,8.6,6e-5)
# Define a wavelength grid (in micron) to downsample the model
obs_wgrid = np.arange(4.8,8.5,0.002)
# The instrument resolving power 
R = 3200

# DISK MODEL

'''Important note: You need to add the molecules in ALPHABETICAL ORDER'''
'''This is just because of how JAX compiles dictionaries.'''
'''So here e.g. we add CO before H2O'''

slab = iris.slab(molecules=['CO', 'H2O'], wlow=4.8, whigh=26.0)

# Distance to source
distance = 120 # pc

# NOTE that we define an ARRAY for the temperature, column density, and area FOR EACH SPECIES.

# Excitation temperatures for each molecule in K
T_ex =  np.array([np.array([600.0]), # for CO  
                  np.array([800.0])]) # for H2O
# column densities in cm^-2
N_mol = np.array([np.array([1e17]), # for CO
                  np.array([1e16])]) # for H2O
# emitting areas in au^2
A_au =  np.array([np.array([2.0]), # for CO
                  np.array([1.0])]) # for H2O
# intrinsic line widths in km/s (line FWHM)
dV =    np.array([np.array([2.0]), # for CO
                  np.array([2.0])]) # for H2O

# model spectrum


slab.setup_disk(distance, T_ex, N_mol, A_au, dV)
slab.setup_grid(fine_wgrid, obs_wgrid, R)
slab.simulate()

# Plot slab.downsampled_flux:
```

<img src="https://github.com/munozcar/IRIS/assets/32044135/b1b92e02-1c82-4144-8398-b557075c2c02"  width="600" height="200">


