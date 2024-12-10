<p align='center'>
  <img src="./src/iris/iris.png" width="250" height="250">
  <br>
</p>


# iris: InfraRed Isothermal Slabs

iris is a Python package that simulates IR molecular line emission from protoplanetary disks. 

## Installation
Get the latest version from PyPi!
```bash
pip install iris-jwst
```
## Requirements
```jax, jaxlib, astroquery, astropy, pandas```

## About

With iris, disk emission lines are modeled using isothermal plane-parallel slabs, with a detailed wavelength-dependent opacity
treatment that accounts for overlapping lines and optical-depth saturation effects. The line profiles can include
turbulent, thermal, and **Keplerian** broadening. 

iris is vectorized and optimized with jax, i.e. the code is dynamically compiled and can (but does not need to) be significantly **sped-up using GPUs**. 
We can generate a full JWST MIRI-MRS model spectrum, including multiple species and temperature gradients in a few millisec.

See the [Wiki](https://github.com/munozcar/iris/wiki) for a detailed explanation and examples.
