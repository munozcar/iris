'''
--------------------------------------------------------------
IRIS: (GPU-accelerated) IR spectrum modeling
--------------------------------------------------------------
Developed by Carlos E. Muñoz-Romero (2023)
'''

import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from iris import spectrum as sp
from iris.moldata import setup_catalog 
import numpy as np

from jax.scipy.signal import fftconvolve

class slab:
    """
        slab: generate an object to model spectra

        :molecules: list of molecules to include in the model (list of strings)
        :wlow: shortest wavelength in micron (float) 
        :whigh:  longest wavelength in micron (float) 
        :path_to_moldata: path to folder with HITRAN molecular data (string) 
        
    """
    def __init__(self, molecules, wlow, whigh, path_to_moldata):
        
        self.molecules = molecules   
        self.catalog, self.levels = setup_catalog(molecules, wlow, whigh, path=path_to_moldata)
        
    def setup_disk(self, distance, T_ex, N_mol, A_au, dV):
        """
        setup_disk: setup the disk physical structure

        :distance: distance to source (float)
        :T_ex: excitation temperature (array)
        :N_mol: column density in cm^-2 (array)
        :A_au: emitting area in au^2 (array)
        :dV: intrinsic line FWHM in km/s (array)
        
        The structure of each physical parameter must be an array of 
        shape (M, N), where N is the number of slabs and M the number of species. 
        For now iris does not support using a different number of slabs 
        for each species.
        
        """
        self.distance =  distance
        self.dV = jnp.array(dV)
        self.T_ex = jnp.array(T_ex)
        self.N_mol = jnp.array(N_mol)
        self.A_au = jnp.array(A_au)
        
    def setup_grid(self, fine_wgrid, obs_wgrid, R):
        """
        setup_grid: setup the wavelength grid

        :fine_wgrid: fine wavelength grid to evaluate the opacity
        :obs_wgrid: wavelength grid used to downsample the model
        :R: the instrumental resolving power
        
        ----------------------------------------------------------------
        To appropiately sample overlapping lines, make sure that the 
        wavelength spacing in fine_wgrid appropiately samples the intrinsic
        line width. Ideally you want a few points per 
        intrinsic width, dlambda<1e-5 micron usually works well for MIRI.
        
        To avoid a nan catastrophe, make sure that obs_wgrid does not span
        outside the range of fine_wgrid. 
        """
        
        self.fine_wgrid = jnp.array(fine_wgrid)
        self.obs_wgrid = jnp.array(obs_wgrid)
        # get bin edges 
        grad = jnp.gradient(self.obs_wgrid)
        lis = list(self.obs_wgrid-grad/2)
        lis.append(self.obs_wgrid[-1] + grad[-1]/2)
        self.hist_wgrid = jnp.array(lis)
        # generate histogram weigths
        self.scale_hist, _ = jnp.histogram(self.fine_wgrid, bins=self.hist_wgrid)
        # resolving power
        self.R = R
        # get fine wavelength spacing
        fine_dw = jnp.max(jnp.gradient(self.fine_wgrid))
        mean_w =  ((self.fine_wgrid[-1]+self.fine_wgrid[0])/2)
        # set up convolution window
        dw_R = mean_w / self.R
        sigma_conv = dw_R/fine_dw/2.355
        self.conv_wind = jax.scipy.stats.norm.pdf(jnp.arange(2000)-1000, scale=sigma_conv)
                
    def convolve(self):
        """
        convolution (internal function called by iris.simulate)
        user shouldn't call this directly
        """
        self.convolved_flux = fftconvolve(self.flux_model, self.conv_wind, mode='same')

    def downsample(self):
        """
        downsampling (internal function called by iris.simulate)
        user shouldn't call this directly
        """
        self.downsampled_flux, _ = jnp.histogram(self.fine_wgrid, bins=self.hist_wgrid, weights=self.convolved_flux)
        self.downsampled_flux = self.downsampled_flux / self.scale_hist
        
    def simulate(self):
        """
        Simulate a convolved and downsampled spectrum
        """
        # generate flux density
        self.flux_model = sp.compute_total_fdens(self.catalog, self.distance, self.A_au, 
                                                 self.T_ex, self.N_mol, self.dV, self.fine_wgrid)
        # convolve with instrument psf
        self.convolve()
        # downsample to grid
        self.downsample()

