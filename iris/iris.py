'''
--------------------------------------------------------------
iris: (GPU-accelerated) IR spectrum modeling
--------------------------------------------------------------
Developed by Carlos E. Muñoz-Romero (2023)
'''
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import spectrum as sp
import numpy as np
from spectres import spectral_resampling_numba
from moldata import setup_catalog 
from scipy.signal import windows
from scipy.ndimage import gaussian_filter1d

class slab:
    def __init__(self, molecules, wlow, whigh, path_to_hitran='./'):
        self.molecules = molecules   
        self.catalog = setup_catalog(molecules, wlow, whigh, path=path_to_hitran)
        
    def setup_disk(self, distance, T_ex, N_mol, A_au, dV):
        """
        setup_disk: setup the disk physical structure

        :distance: distance to source (float)
        :T_ex: fine wavelength grid to evaluate tau (array)
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
        :obs_wgrid: the wavelength grid used to downsample the model
        :R: the instrument resolving power (lambda/dlambda)
        
        ----------------------------------------------------------------
        To appropiately sample overlapping lines, make sure that the 
        wavelength spacing in fine_wgrid is smaller than the equivalent
        width of 1 km/s in microns. Ideally you want a few points per 
        intrinsic width, and a fine dlambda<1e-5 micron usually works well.
        
        To avoid a nan catastrophe, make sure that obs_wgrid does not span
        outside the range of fine_wgrid. It is very expensive to assert this
        at every iteration, hence it is the responsibility of the user to do so.
                
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
        self.conv_wind = jax.scipy.stats.norm.pdf(jnp.arange(500)-250, scale=sigma_conv)
                
    def convolve(self):
        '''
        convolution routine (internal function called by iris.simulate)
        user should not call this directly
        '''
        self.convolved_flux = jnp.convolve(self.flux_model, self.conv_wind, mode='same')

    def downsample(self):
                '''
        downsampling routine (internal function called by iris.simulate)
        user should not call this directly
        '''
        self.downsampled_flux, _ = jnp.histogram(self.fine_wgrid, bins=self.hist_wgrid, weights=self.convolved_flux)
        self.downsampled_flux = self.downsampled_flux / self.scale_hist
        
    def simulate(self):
        # generate flux density
        self.flux_model = sp.compute_total_fdens(self.catalog, self.distance, self.A_au, 
                                                 self.T_ex, self.N_mol, self.dV, self.fine_wgrid)
        # convolve with instrument psf
        self.convolve()
        # downsample to grid
        self.downsample()
        


