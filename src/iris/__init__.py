'''
--------------------------------------------------------------
IRIS: (GPU-accelerated) IR spectrum modeling
--------------------------------------------------------------
Developed by Carlos E. Romero-Mirza (2024)
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
        
    def setup_disk(self, distance, T_ex, N_mol, A_au, dV, inc=0.0, M_star=1.0, r_in=0.1):
        """
        setup_disk: setup the disk physical structure

        :distance: distance to source (float)
        :T_ex: excitation temperature (array)
        :N_mol: column density in cm^-2 (array)
        :A_au: emitting area in au^2 (array)
        :dV: intrinsic line FWHM in km/s (array)

        :inc: disk inclination in degree (float), face-on = 0, experimental
        :M_star: stellar mass in Msun (float), experimental
        :r_in: inner gas radius in au (array), experimental. An ARRAY of the inner emission 
               radius for EACH molecule included in the model
        
        The structure of each physical parameter must be an array of 
        shape (M, N), where N is the number of slabs and M the number of species. 
        
        For now iris does not support using a different number of temperature components 
        for each species modeled simultaneously.
        
        """
        self.distance =  distance
        self.dV = jnp.array(dV)
        self.T_ex = jnp.array(T_ex)
        self.N_mol = jnp.array(N_mol)
        self.A_au = jnp.array(A_au)
        
        self.inc = inc*np.pi/180.0 * jnp.ones_like(self.A_au)
        self.M_star = M_star * jnp.ones_like(self.A_au)
        self.r_in = jnp.ones_like(self.A_au)
        self.r_in = self.r_in.at[:,0].set(r_in)
        for i in range(1, self.A_au.shape[1]):
            self.r_in = self.r_in.at[:,i].set( (self.A_au[:, i-1]/jnp.pi + self.r_in[:,i-1]**2)**0.5 )

            
        
    def setup_grid(self, fine_wgrid, obs_wgrid, R):
        """
        setup_grid: setup the wavelength grid

        :fine_wgrid: fine wavelength grid to evaluate the opacity
        :obs_wgrid: wavelength grid used to downsample the model
        :R: the instrumental resolving power
        
        ----------------------------------------------------------------
        To correctly sample overlapping lines, make sure that the 
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

    def simulate_keplerian(self):
        """
        Simulate a convolved and downsampled spectrum including Keplerian line profiles
        Requires inc, M_star, and r_in to be set up first.
        Experimental
        """
        # generate flux density
        self.flux_model = sp.compute_total_fdens_keplerian(self.catalog, self.distance, self.A_au, 
                                         self.T_ex, self.N_mol, self.dV, self.r_in,
                                         self.M_star, self.inc, self.fine_wgrid)
        # convolve with instrument psf
        self.convolve()
        # downsample to grid
        self.downsample()

