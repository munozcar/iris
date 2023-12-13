'''
--------------------------------------------------------------
IRIS: (GPU-accelerated) IR spectrum modeling

Evaluation of optical depths and intensities is jit-enabled and
vectorized. 
--------------------------------------------------------------
Developed by Carlos E. Muñoz-Romero (2023)
'''
import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
import astropy.constants as const
import jax.numpy as jnp
import numpy as np
from functools import partial

# -------------------- Global constants ----------------------
# for more efficient unit handling
# au to cm
aucm = const.au.to(u.cm)
# pc to cm
pccm = const.pc.to(u.cm)
# squared arcsec to sr
arcsec2_to_sr = ((u.arcsec**2).to(u.sr))
# c in micron/sec
c_musec = (const.c.to(u.micron/u.s)).value
# c in cm/sec
c_cmsec = (const.c.cgs).value
# km/s to micron/s
kms_to_mus = ((1*u.km/u.s).to(u.micron/u.s)).value
# plack const. in cgs
h_cgs = (const.h.cgs).value
# boltzmann const. in cgs
k_B_cgs = (const.k_B.cgs).value    
# c/micron
micron_to_hz = (const.c/(1*u.micron)).to(u.Hz)
# mass of the proton
mp_cgs = const.m_p.cgs.value
# ----------------------------------------------------------

# --------------- Main modeling code -----------------------

def J_profile(wavelength, t_ex): 
    """
    J_profile: full (blackbody) Plack function

    :wavelength: wavelength in micron
    :t_ex: temperature in K
    
    :return: blackbody flux density in erg cm-2 s-1 Hz-1 sr-1
    """ 
    nu = c_cmsec/(wavelength*1e-4)
    return (2*h_cgs*nu**3)/(c_cmsec**2) * (jnp.exp(h_cgs*nu/(k_B_cgs*t_ex)) - 1)**(-1)
    
J_profile = jax.jit(J_profile)

def dv_to_dlam(dv, lam_rest):
    """
    dv_to_dlam: convert velocity element in km/s to wavelength element in micron,
                given a reference wavelength
    :dv: elocity element in km/s
    :lam_rest: reference wavelength in micron
    
    returns: dlam in micron
    """
    nu_rest = c_musec/lam_rest
    lam =  -c_musec / (((dv*kms_to_mus*nu_rest)/c_musec) - nu_rest)
    return lam - lam_rest
dv_to_dlam = jax.jit(dv_to_dlam)

def evaluate_line_tau(fine_wgrid, sigma_lam, tau_cen, line_w):
    """
    evaluate_line_tau: get line optical depth profile 
    
    :fine_wgrid: fine wavelength grid to evaluate tau 
    :sigma_lam: intrinsic line (gaussian) width in micron
    :tau_cen: optical depth at line center
    :line_w: wavelenbgth at line center in micron
    
    returns: tau (unitless)
    """
    # Gaussian profile
    return tau_cen * jnp.exp(-((fine_wgrid-line_w)**2)/(2*sigma_lam**2) )
evaluate_line_tau = jax.jit(evaluate_line_tau)

def compute_tau_grid(line_catalog, fine_wgrid, dv, t_ex, n_mol):
    """
    compute_tau_grid: get optical depth profile for all lines in catalog
    
    :line_catalog: HITRAN catalog 
    :fine_wgrid: fine wavelength grid to evaluate tau 
    :dv: intrinsic line FWHM in km/s
    :t_ex: excitation temperature in K
    :n_mol: column density in cm^-2
    
    returns: tau (unitless)
    """
    # partition function
    q_sum = jnp.interp(t_ex, line_catalog['Qt'], line_catalog['Qv'])
    # population levels
    x_low = line_catalog['glow'] * jnp.exp(-line_catalog['elow'] / t_ex) / q_sum
    x_up = line_catalog['gups'] * jnp.exp(-line_catalog['eups'] / t_ex) / q_sum
    # optical depth
    tau_amp = (jnp.log(2) / jnp.pi)**0.5 * (line_catalog['aijs'] * n_mol * (line_catalog['ws']*1e-4)**3) / (4*jnp.pi*dv*(1e5))
    tau_lvl = x_low * line_catalog['gups']/line_catalog['glow'] - x_up
    tau_cen = tau_amp * tau_lvl
    # intrinsic width in micron
    sigma_lam = dv_to_dlam(dv/2.355, fine_wgrid)
    # map over all lines
    func = partial(evaluate_line_tau, fine_wgrid, sigma_lam)
    tau_grid = jax.vmap(func)(tau_cen, line_catalog['ws'])
    
    # sum opacities
    return  jnp.sum(tau_grid, axis=0)
compute_tau_grid = jax.jit(compute_tau_grid)

def compute_total_tau(catalog, fine_wgrid, dv, t_ex, n_mol):
    """
    compute_tau_grid: get optical depth profile for all molecules
    
    :catalog: HITRAN catalogs
    :fine_wgrid: fine wavelength grid to evaluate tau 
    :dv: intrinsic line FWHM in km/s
    :t_ex: excitation temperature in K
    :n_mol: column density in cm^-2
    
    returns: tau (unitless)
    """
    keys = list(catalog.keys())
    total_tau = []
    # loop over each species
    for i in range(len(catalog)):
        # map over disk grid
        func = partial(compute_tau_grid, catalog[keys[i]], fine_wgrid)
        total_tau.append(jax.vmap(func)(dv[i], t_ex[i], n_mol[i]))
    return jnp.array(total_tau)

compute_total_tau = jax.jit(compute_total_tau)

def compute_fdens(distance, fine_wgrid, total_tau, A_au, t_ex):
    """
    compute_fdens: get flux density for one species
    
    :distance: distance to source in pc
    :fine_wgrid: fine wavelength grid to evaluate tau 
    :total_tau: tau as function of wavelength
    :A_au: emitting area in au^2
    :t_ex: excitation temperature in K
    
    returns: flux density (erg cm-2 s-1 Hz-1)
    """
    # radius assuming circular area
    eq_radius = (A_au/jnp.pi)**0.5
    # line intensities accounting for saturation at line center
    return jnp.pi * (eq_radius/distance)**2 * ((aucm / pccm) ** 2) * J_profile(fine_wgrid, t_ex) * (1 - jnp.exp(-total_tau))
compute_fdens = jax.jit(compute_fdens)

def compute_total_fdens(catalog, distance, A_au, t_ex, n_mol, dv, fine_wgrid):
    """
    compute_total_fdens: get total flux density in Jy
    
    :catalog: HITRAN catalog (dict)
    :distance: distance to source in pc
    :A_au: emitting area in au^2
    :t_ex: excitation temperature in K
    :n_mol: column density in cm^-2
    :dv: intrinsic line FWHM in km/s
    :fine_wgrid: fine wavelength grid to evaluate tau 
   
    returns: flux density (Jy)
    """
    fdens = fine_wgrid*0.0
    # evaluate total optical depth profile
    total_tau = compute_total_tau(catalog, fine_wgrid, dv, t_ex, n_mol)
    # loop over species
    for i in range(len(catalog)):
        # map over disk grid
        func = partial(compute_fdens, distance, fine_wgrid)
        fdens_i = jax.vmap(func)(total_tau[i], A_au[i], t_ex[i])
        fdens += jnp.sum(fdens_i, axis=0) 
    return fdens * 1e23
compute_total_fdens = jax.jit(compute_total_fdens)
