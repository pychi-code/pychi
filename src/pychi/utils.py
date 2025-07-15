# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 09:42:48 2025

@author: Thibault Voumard

This module contains utility functions that either make sense to keep out of
other modules or are annoying to implement otherwise, although could be at a
later stage.
"""
import math
import numpy as np
from scipy.constants import c

def betas_from_dispersion(wavelength, n, disp=0, disp_slope=0):
    """
    Compute the corresponding set of betas from typical dispersion values
    provided by fiber manufacturers. Should eventually belong to waveguide
    module.

    Parameters
    ----------
    wavelength : float
        Wavelength at which the dispersion is specified.
    n : float
        Refractive index at the specified wavelength.
    disp : float, optional
        Dispersion at the specified wavelength. Default is 0.
    disp_slope : float, optional
        Dispersion slope at the specified wavelength. Default is 0.

    Returns
    -------
    list
        Beta coefficients corresponding to the given dispersion.

    """
    beta_0 = n*2*np.pi/wavelength
    beta_1 = n/c
    beta_2 = -disp*wavelength**2/2/np.pi/c
    beta_3 = -wavelength**3/(2*np.pi*c)**2*(2*disp-wavelength*disp_slope)
    return [beta_0, beta_1, beta_2, beta_3]

def betas_from_dispersion_non_SI(wavelength, n, disp=0, disp_slope=0):
    """
    Compute the corresponding set of betas from typical dispersion values
    provided by fiber manufacturers in their non SI but more common units,
    i.e. ps/(nm km) and ps/(nm**2 km). Should eventually belong to waveguide
    module.

    Parameters
    ----------
    wavelength : float
        Wavelength at which the dispersion is specified.
    n : float
        Refractive index at the specified wavelength.
    disp : float, optional
        Dispersion at the specified wavelength, in ps/(nm km). Default is 0.
    disp_slope : float, optional
        Dispersion slope at the specified wavelength, in ps/(nm**2 km). Default
        is 0.

    Returns
    -------
    list
        Beta coefficients corresponding to the given dispersion.

    """
    disp = 1e6*disp
    disp_slope = 1e-3*disp_slope
    beta_0 = n*2*np.pi/wavelength
    beta_1 = n/c
    beta_2 = -disp*wavelength**2/2/np.pi/c
    beta_3 = -wavelength**3/(2*np.pi*c)**2*(2*disp-wavelength*disp_slope)
    return [beta_0, beta_1, beta_2, beta_3]

def n_eff_from_betas(freq, wavelength, betas):
    """
    Compute the effective refractive index from a set of beta coefficients.
    Should eventually belong to waveguide module.

    Parameters
    ----------
    freq : array of float
        Frequency axis over which the refractive index will be defined.
    wavelength : float
        Wavelength at which the beta coefficients are defined.
    betas : list of float
        List of the beta coefficients.

    Returns
    -------
    n_eff : array of float
        Effective refractive indices computed from the beta coefficients.

    """
    omega = 2*np.pi*freq
    k = np.zeros(len(freq), dtype='float64')
    for i, beta in enumerate(betas):
        k += beta*(omega - 2*np.pi*c/wavelength)**i/math.factorial(i)
    n_eff = k*c/omega
    return n_eff
