<p align='center'>
  <img src="./src/iris/iris.png" width="250" height="250">
  <br>
</p>


# iris: InfraRed Isothermal Slabs

iris is a Python package that simulates IR molecular line emission from protoplanetary disks. 

## Installation
Get the latest version from PyPi

pip install iris-jwst

## Requirements
jax, jaxlib, astroquery, astropy, pandas

## About

The emission lines are modeled using isothermal slabs, with a detailed wavelength-dependent opacity
treatment that accounts for overlapping lines and saturation effects. 

iris is vectorized and optimized with jax, i.e. the code is dynamically compiled and can (but does not need to) be significantly sped-up using GPUs. 
e.g. we can generate a full JWST MIRI MRS model spectrum including multiple species in < 5 ms

See the wiki for a more detailed explanation 
