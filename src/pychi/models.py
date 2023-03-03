# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:24:21 2022

@author: voumardt
"""
import numba
import numpy as np
import pyfftw
from scipy.constants import c
import time as timing
import matplotlib.pyplot as plt

class Model():
    """
    Parent class for the different nonlinear Schrödinger equations. Provide
    general utility, optimization and setup functions for the physics
    happening in the child classes.
    """
    def __init__(self, waveguide, light):
        """
        Construct model class. Some attributes are initialized in other methods.

        Parameters
        ----------
        waveguide : Waveguide
            Object containing the material parameters.
        light : Light
            Object containing the light parameters.
        """
        self.waveguide = waveguide
        self.light = light
        self.half_pts = self.waveguide.t_pts//2
        
    def _setup_aaspm(self):
        """
        Setup storage, ffts and selection array for anti-aliasing spm
        """
        # padded fft
        self.spm_pad_factor = 2 # On the Elimination of Aliasing in a Finite-Difference Schemes by Filtering High Wavenumber
        self._fft_a_aaspm = pyfftw.empty_aligned(self.spm_pad_factor*self.waveguide.t_pts, dtype='complex128')
        self._fft_b_aaspm = pyfftw.empty_aligned(self.spm_pad_factor*self.waveguide.t_pts, dtype='complex128')
        self._fft_c_aaspm = pyfftw.empty_aligned(self.spm_pad_factor*self.waveguide.t_pts, dtype='complex128')
        self._fft_d_aaspm = pyfftw.empty_aligned(self.spm_pad_factor*self.waveguide.t_pts, dtype='complex128')
        self.fft_spm = pyfftw.FFTW(self._fft_a_aaspm, self._fft_b_aaspm)
        self.ifft_spm = pyfftw.FFTW(self._fft_c_aaspm, self._fft_d_aaspm, direction='FFTW_BACKWARD')
        
        # selection arrays
        self.selection_array_spm = np.zeros(self.spm_pad_factor*self.waveguide.t_pts, dtype=bool)
        self.selection_array_spm[:self.half_pts] = True
        self.selection_array_spm[-self.half_pts:] = True
        self.storage_array_spm = np.zeros(self.spm_pad_factor*self.waveguide.t_pts, dtype='complex128')
        
        # Raman term
        self.raman_effect = np.zeros(self.spm_pad_factor*self.waveguide.t_pts, dtype='complex128')
        self.raman_effect[:self.half_pts] = self.waveguide.raman_effect[:self.half_pts]
        self.raman_effect[-self.half_pts:] = self.waveguide.raman_effect[self.half_pts:]
        self.raman_effect += (1 - self.waveguide.raman_fraction)
        self.raman_effect *= 3*self.spm_pad_factor**2
    
    def _setup_aashg(self):
        """
        Setup storage, selection array and exponents for anti-aliasing chi 2
        frequency conversion
        """
        # padded fft
        self.shg_pad_exp = np.ceil(np.log2(3/2 + self.waveguide.center_omega/2/np.amax(self.waveguide.rel_omega)))
        self.shg_pts_pad = int(self.waveguide.t_pts*2**self.shg_pad_exp)
        self._fft_a_shg = pyfftw.empty_aligned(self.shg_pts_pad, dtype='complex128')
        self._fft_b_shg = pyfftw.empty_aligned(self.shg_pts_pad, dtype='complex128')
        self._fft_c_shg = pyfftw.empty_aligned(self.shg_pts_pad, dtype='complex128')
        self._fft_d_shg = pyfftw.empty_aligned(self.shg_pts_pad, dtype='complex128')
        self.fft_shg = pyfftw.FFTW(self._fft_a_shg, self._fft_b_shg)
        self.ifft_shg = pyfftw.FFTW(self._fft_c_shg, self._fft_d_shg, direction='FFTW_BACKWARD')
        
        # selection arrays
        self.selection_array_shg = np.zeros(self.shg_pts_pad, dtype=bool)
        self.selection_array_shg[:self.half_pts] = True
        self.selection_array_shg[-self.half_pts:] = True
        self.storage_array_shg = np.zeros(self.shg_pts_pad, dtype='complex128')
        
        # frequency shifting term
        self.shg_time = np.linspace(np.amin(self.waveguide.time), np.amax(self.waveguide.time), self.shg_pts_pad)
        self.exponent_sfg = numbaexp(1j*self.waveguide.center_omega*self.shg_time)*2**self.shg_pad_exp
        self.exponent_dfg = numbaconj(self.exponent_sfg)
    
    def _setup_aathg(self):
        """
        Setup storage, selection array and exponents for anti-aliasing chi 3
        frequency conversion
        """
        # padded fft
        self.thg_pad_exp = np.ceil(np.log2(2 + self.waveguide.center_omega/np.amax(self.waveguide.rel_omega)))
        self.thg_pts_pad = int(self.waveguide.t_pts*2**self.thg_pad_exp)
        self._fft_a_thg = pyfftw.empty_aligned(self.thg_pts_pad, dtype='complex128')
        self._fft_b_thg = pyfftw.empty_aligned(self.thg_pts_pad, dtype='complex128')
        self._fft_c_thg = pyfftw.empty_aligned(self.thg_pts_pad, dtype='complex128')
        self._fft_d_thg = pyfftw.empty_aligned(self.thg_pts_pad, dtype='complex128')
        self.fft_thg = pyfftw.FFTW(self._fft_a_thg, self._fft_b_thg)
        self.ifft_thg = pyfftw.FFTW(self._fft_c_thg, self._fft_d_thg, direction='FFTW_BACKWARD')
        
        # selection arrays
        self.selection_array_thg = np.zeros(self.thg_pts_pad, dtype=bool)
        self.selection_array_thg[:self.half_pts] = True
        self.selection_array_thg[-self.half_pts:] = True
        self.storage_array_thg = np.zeros(self.thg_pts_pad, dtype='complex128')
        
        # frequency shifting terms
        self.thg_time = np.linspace(np.amin(self.waveguide.time), np.amax(self.waveguide.time), self.thg_pts_pad)
        self.exponent_thg = numbaexp(2j*self.waveguide.center_omega*self.thg_time)*(2**self.thg_pad_exp)**2
        
    
    """
    Nonlinear operations
    """
    def _aaspm(self, field_f):
        """
        Compute anti-aliasing self-phase modulation term with Raman effect.

        Parameters
        ----------
        field_f : array
            Field in frequency domain.

        Returns
        -------
        array
            Anti-aliased SPM term in frequency domain.
        """
        self.storage_array_spm[:self.half_pts] = field_f[:self.half_pts]
        self.storage_array_spm[-self.half_pts:] = field_f[self.half_pts:]
        field_t_pad = numbacopy(self.ifft_spm(self.storage_array_spm))
        int_f = self.fft_spm(numbaabs2(field_t_pad))
        spm = self.ifft_spm(int_f*self.raman_effect)*field_t_pad
        return self.fft_spm(spm)[self.selection_array_spm]

    def _aash(self, field_f):
        """
        Compute combined anti-aliasing sum frequency generation and difference
        frequency generation

        Parameters
        ----------
        field_f : array
            Field in frequency domain.

        Returns
        -------
        array
            Anti-aliased SFG and DFG in frequency domain.
        """
        self.storage_array_shg[:self.half_pts] = field_f[:self.half_pts]
        self.storage_array_shg[-self.half_pts:] = field_f[self.half_pts:]
        sht = numbasht(self.ifft_shg(self.storage_array_shg), self.exponent_sfg)
        return self.fft_shg(sht)[self.selection_array_shg] 

    def _aathg(self, field_f):
        """
        Compute combined anti-aliasing triple and inverse triple harmonic.

        Parameters
        ----------
        field_f : array
            Field in time domain.

        Returns
        -------
        array
            Anti-aliased triple and inverse triple harmonic in frequency domain.
        """
        self.storage_array_thg[:self.half_pts] = field_f[:self.half_pts]
        self.storage_array_thg[-self.half_pts:] = field_f[self.half_pts:]
        field_t_mult = numbathg(self.ifft_thg(self.storage_array_thg), self.exponent_thg)
        return self.fft_thg(field_t_mult)[self.selection_array_thg]

    
    """
    Propagation functions
    """
    def nonlinear_term(self, field):
        """
        Nonlinear term, implemented in child classes for different interaction
        types.
        """
        raise NotImplementedError
    
    def dispersion_step(self, field, dz):
        """
        Compute dispersion step with field in frequency domain.

        Parameters
        ----------
        field : array
            Field in frequency domain.
        dz : float
            Step size.

        Returns
        -------
        array
            Dispersion affected field in frequency domain.
        """
        return numbaexp(-1j*self.waveguide.k*dz)*field


"""
Model classes
"""
class Spm(Model):
    """
    Self phase modulation only physical model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aaspm()
        self._recommended_solver = 'ERK4IP'
        
    def nonlinear_term(self, field_f):
        """
        Nonlinear function with spm

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        tot_spm = self._aaspm(field_f)
        return self.waveguide.rhs_prefactor*tot_spm*self.waveguide.chi_3/4


class Chi3(Model):
    """
    Triple sum-frequency physical model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aathg()
    
    def nonlinear_term(self, field_f):
        """
        Nonlinear function with spm and chi 3

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        thg_term = self._aathg(field_f)
        return self.waveguide.rhs_prefactor*thg_term*self.waveguide.chi_3/4
        

class SpmChi3(Model):
    """
    Self phase modulation and triple harmonic physical model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aaspm()
        self._setup_aathg()
    
    def nonlinear_term(self, field_f):
        """
        Nonlinear function with spm and chi 3

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        tot_spm = self._aaspm(field_f)
        thg_term = self._aathg(field_f)
        return self.waveguide.rhs_prefactor*(thg_term + tot_spm)*self.waveguide.chi_3/4


class Chi2(Model):
    """
    Sum frequency generation and difference frequency generation only physical
    model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aashg()

    def nonlinear_term(self, field_f):
        """
        Nonlinear function with chi 2

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        shg_term = self._aash(field_f)
        return self.waveguide.rhs_prefactor*shg_term*self.waveguide.chi_2/2
    

class SpmChi2(Model):
    """
    Self phase modulation, sum and difference frequency
    generation physical model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aaspm()
        self._setup_aashg()

    def nonlinear_term(self, field_f):
        """
        Nonlinear function with spm and chi 2

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        tot_spm = self._aaspm(field_f)
        shg_term = self._aash(field_f)
        return self.waveguide.rhs_prefactor*(tot_spm*self.waveguide.chi_3/4 + shg_term*self.waveguide.chi_2/2)


class Chi2Chi3(Model):
    """
    Triple-sum, sum and difference frequency
    generation physical model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aashg()
        self._setup_aathg()

    def nonlinear_term(self, field_f):
        """
        Nonlinear function with chi 2 and chi 3

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        shg_term = self._aash(field_f)
        thg_term = self._aathg(field_f)
        return self.waveguide.rhs_prefactor*(thg_term*self.waveguide.chi_3/4 + shg_term*self.waveguide.chi_2/2)


class SpmChi2Chi3(Model):
    """
    Self phase modulation, triple harmonic, sum and difference frequency
    generation physical model.
    """
    def __init__(self, waveguide, light):
        """
        Class constructor.

        Parameters
        -------
        waveguide : Waveguide
            Object with the material properties of the propagation medium.

        light : Light
            Object containing the optical field.
        """
        Model.__init__(self, waveguide, light)
        self._setup_aaspm()
        self._setup_aashg()
        self._setup_aathg()
        
    def nonlinear_term(self, field_f):
        """
        Nonlinear function with spm, chi 2 and chi 3

        Parameters
        -------
        field_f : array
            Analytical envelope of the electric field in frequency domain.

        Returns
        -------
        array
            Nonlinear evolution term.
        """
        tot_spm = self._aaspm(field_f)
        thg_term = self._aathg(field_f)
        shg_term = self._aash(field_f)
        return self.waveguide.rhs_prefactor*((thg_term + tot_spm)*self.waveguide.chi_3/4 + shg_term*self.waveguide.chi_2/2)
    

"""
Optimized functions
"""
### Numba exponential
@numba.njit([numba.complex128[:](numba.complex128[:]),
             numba.complex64[:](numba.complex64[:])])
def numbaexp(x):
    """Compute numba optimized exponential of input vector."""
    return np.exp(x)

### Numba absolute value squared
@numba.vectorize([numba.float64(numba.complex128),
                  numba.float32(numba.complex64)])
def numbaabs2(x):
    """Compute numba optimized absolute norm square of input vector."""
    return x.real**2 + x.imag**2

### Numba copy
@numba.njit([numba.complex128[:](numba.complex128[:]),
             numba.complex64[:](numba.complex64[:])])
def numbacopy(x):
    """Create a numba optimized copy of a vector."""
    return np.copy(x)

### Numba conjugation
@numba.njit([numba.complex128[:](numba.complex128[:]),
             numba.complex64[:](numba.complex64[:])])
def numbaconj(x):
    """Compute numba optimized complex conjugate of a vector."""
    return np.conj(x)

### Numba shg
@numba.vectorize([numba.complex128(numba.complex128, numba.complex128),
                  numba.complex64(numba.complex64, numba.complex64)])
def numbasht(x, y):
    """Compute numba optimized sh term."""
    return x*x*y  + 2*(x.real**2 + x.imag**2)*(y.real - 1j*y.imag)

### Numba thg
@numba.vectorize([numba.complex128(numba.complex128, numba.complex128),
                  numba.complex64(numba.complex64, numba.complex64)])
def numbathg(x, y):
    """Compute numba optimized th term."""
    return x*x*x*y + 3*(x.real**2 + x.imag**2)*(x.real - 1j*x.imag)*(y.real - 1j*y.imag)
