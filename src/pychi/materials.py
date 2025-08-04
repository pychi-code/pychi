# -*- coding: utf-8 -*-
from inspect import signature
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c
eps_0 = 8.8541878128e-12

class Waveguide():
    """
    Waveguide class. Contains all material related information, i.e. the
    refractive index curve versus frequency, the Raman parameters, the
    nonlinear coefficients, the effective area and length. This object
    should be the first object instantiated, as the time/frequency axes of
    the simulation are derived here from the available refractive
    index data.

    Parameters
    ----------
    frequency : array
        Frequency axis on which the effective refractive index is given.
    n_eff : float, vector, array or callable
        Effective refractive index of the material.
    chi_2 : float, vector, array or callable
        Order 2 material nonlinearity. 
    chi_3 : float, vector, array or callable
        Order 3 material nonlinearity.
    effective_area : float
        Effective area at the pump wavelength.
    length : float
        Length of the material.
    raman_fraction : float
        Fractional contribution of the Raman effect to the Kerr effect.
        The default is 0.18.
    raman_tau_1 : float
        Raman effect primary time scale. The default is 0.0122e-12.
    raman_tau_2 : float
        Raman effect secondary time scale. The default is 0.032e-12.
    t_pts : float, optional
        Number of points in the time and frequency axes, will be rounded
        to nearest higher power of 2. The default is 2**14.
    """
    def __init__(self, frequency, n_eff, chi_2, chi_3,
                 effective_area, length, raman_fraction=0.18,
                 raman_tau_1=0.0122e-12, raman_tau_2=0.032e-12,
                 t_pts=2**14):
        self.z = 0
        self.t_pts = int(2**np.ceil(np.log2(t_pts)))
        self.length = length
        self.data_freq = frequency
        self.center_freq, self.freq = self._compute_frequency_axis()
        self.center_omega, self.omega, self.rel_omega = self._compute_omega_axis()
        self.time, self.delta_t = self._compute_time_axis()
        
        self.n_eff = n_eff
        
        self.chi_2 = chi_2
        self.chi_3 = chi_3
        self.eff_area = effective_area
        
        # Standard Raman response for fibers
        self.raman_fraction = raman_fraction
        self.raman_tau_1 = raman_tau_1
        self.raman_tau_2 = raman_tau_2
        self.raman_effect = self._compute_raman_effect()
    
    @property
    def n_eff(self):
        """Get the effective refractive index at position z."""
        return self._n_eff(self.z)
    
    @n_eff.setter
    def n_eff(self, n_eff):
        """
        Set the effective refractive index. Change any float, vector or
        array provided by the user into a z-callable function. A version
        interpolated in frequency is also kept for other operations.
        Beta coefficients are computed at this stage - they are
        solely used for plotting purpose.
        """
        self._n_eff = self._make_callable(n_eff)
        self.n_eff_inter = self._interpolate_n_eff()
        self.betas = self.compute_betas()
    
    @property
    def chi_2(self):
        """Get the quadratic nonlinear coefficient at position z."""
        return self._chi_2(self.z)*np.sqrt(self._eff_area(0)/self._eff_area(self.z))
    
    @chi_2.setter
    def chi_2(self, chi_2):
        """
        Set the quadractic nonlinear coefficient. Change any float,
        vector or array provided by the user into a z-callable
        function.
        """
        self._chi_2 = self._make_callable(chi_2)
        
    @property
    def chi_3(self):
        """Get the cubic nonlinear coefficient at position z."""
        return self._chi_3(self.z)*self._eff_area(0)/self._eff_area(self.z)
    
    @chi_3.setter
    def chi_3(self, chi_3):
        """
        Set the quadractic nonlinear coefficient. Change any float,
        vector or array provided by the user into a z-callable
        function.
        """
        self._chi_3 = self._make_callable(chi_3)
        
    @property
    def eff_area(self):
        """Get the effective area at position z."""
        return self._eff_area(self.z)
    
    @eff_area.setter
    def eff_area(self, eff_area):
        """
        Set the effective area. Change any float,
        vector or array provided by the user into a z-callable
        function.
        """
        self._eff_area = self._make_callable(eff_area)
    
    @property
    def k(self):
        """Get the wavevectors at position z."""
        return np.fft.fftshift(self.n_eff*self.omega/c)
        
    @property
    def rhs_prefactor(self):
        """Get the GNLSE right-hand-side prefactor at position z."""
        return -1j*np.fft.fftshift(self.omega/(2*self.n_eff*c))
    
    def _make_callable(self, var):
        """
        Transform an input vector, array, float or callable into a z-callable,
        allowing the user to provide z-dependant quantities.
        
        Note that a provided callable should have as arguments exactly at most
        'z' and 'freq', e.g. var(z, freq), var(freq, z), var(z) or var(freq).
        A bit wonky, but couldn't figure out a better way to do that. Another
        alternative would be to implement derived classes with overwritten
        properties, which is not very elegant either. Actually found out that
        singledispatch from functools is probably the way to go here.
        
        Note that a provided array should have first dimension coinciding with
        the frequency axis given along the refractive index data. The second
        dimension is assumed to be evenly distributed along the whole length
        of the waveguide.
        
        Note that if a vector is provided it will be assumed that it describes
        a z dependent quantity, unless it has the same dimension as the
        frequency axis. Wavelength dependent only quantities can also be
        provided through callables.

        Parameters
        ----------
        var : float, vector, array or callable
            Some simulation parameter.

        Returns
        -------
        function
            A z-callable version of the provided variable.
        """
        # Case when the argument is callable
        if callable(var):
            return self._parse_callable(var)
        # Case when the argument is an array or a vector
        elif hasattr(var, '__len__'):
            return self._parse_array(var)
        # Otherwise (float)
        else:
            return lambda _: var
            
    def _parse_array(self, var):
        """
        Parse the correct arguments for a callable parameter.

        Parameters
        ----------
        var : array
            A one or two dimensional array representing z and/or frequency
            dependence of a parameter.

        Returns
        -------
        function
            A z-callable function.
        """
        # Case of an array
        if len(np.shape(var)) == 2:
            arr_inter = interp1d(self.data_freq, var.T, kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')
            arr = arr_inter(self.freq).T
            z = np.linspace(0, self.length, np.shape(arr)[-1])
            arr_inter = interp1d(z, arr, kind='cubic', bounds_error=False,
                                 fill_value='extrapolate')
            return arr_inter
        # Case of a vector
        elif len(np.shape(var)) == 1:
            if len(var) == len(self.data_freq):
                arr_inter = interp1d(self.data_freq, var, kind='cubic',
                                     bounds_error=False, fill_value='extrapolate')
                arr = arr_inter(self.freq)
                return lambda _: arr
            else:
                z = np.linspace(0, self.length, len(var))
                var_inter = interp1d(z, var, kind='cubic', bounds_error=False,
                                     fill_value='extrapolate')
                return var_inter
        else:
            raise ValueError
                
    def _parse_callable(self, var):
        """
        Parse the correct arguments for a callable parameter.

        Parameters
        ----------
        var : callable
            A z or frequency (or both) dependent callable.

        Returns
        -------
        function
            A z-callable function.
        """
        # Get funtion signature
        z_flag = False
        freq_flag = False
        order_flag = False
        args = signature(var).parameters
        for arg in args:
            if arg == 'z':
                z_flag = True
            if arg == 'freq':
                freq_flag = True
                if z_flag:
                    order_flag = True
        
        # Reorder arguments and evaluate the frequency axis
        if z_flag:
            if freq_flag:
                if order_flag:
                    return lambda _: var(_, self.freq)
                else:
                    return lambda _: var(self.freq, _)
            else:
                return var
        else:
            if freq_flag:
                return lambda _: var(self.freq)
            else:
                return lambda _: var
    
    def _compute_frequency_axis(self):
        """
        Define a frequency axis covering exactly the data range available.
        """
        min_freq = np.amin(self.data_freq)
        max_freq = np.amax(self.data_freq)
        center_freq = min_freq/2 + max_freq/2
        freq = np.linspace(min_freq, max_freq, self.t_pts)
        return center_freq, freq
        
    def _compute_omega_axis(self):
        """
        Compute the angular frequency axis associated to the frequency axis
        """
        center_omega = 2*np.pi*self.center_freq
        omega = 2*np.pi*self.freq
        rel_omega = omega - center_omega
        return center_omega, omega, rel_omega
    
    def _compute_time_axis(self):
        """
        Compute the time axis associated to the frequency axis
        """
        time = np.fft.fftshift(np.fft.fftfreq(self.t_pts, np.mean(np.diff(self.freq))))
        delta_t = np.mean(np.diff(time))
        return time, delta_t
    
    def _interpolate_n_eff(self):
        """
        Interpolate the refractive index data on the simulation frequency axis
        """
        n_eff_inter = interp1d(self.freq, self.n_eff, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        return n_eff_inter
    
    def _compute_raman_effect(self):
        """
        Compute the Raman term for the simulation frequency axis
        """
        raman_time = (self.raman_tau_1**2 + self.raman_tau_2**2)
        raman_time = raman_time/self.raman_tau_1/self.raman_tau_2**2
        raman_time *= np.exp(-np.abs(self.time)/self.raman_tau_2)*np.sin(self.time/self.raman_tau_1)
        raman_time[self.time < 0] = 0
        raman_effect = np.conjugate(self.t_pts*np.fft.ifft(np.fft.fftshift(raman_time)))
        raman_effect *= self.delta_t*self.raman_fraction
        return raman_effect
    
    def compute_betas(self, wavelength=None, order=6):
        """
        Compute beta coefficients at a given wavelength and order.

        Parameters
        ----------
        wavelength : float, optional
            Wavelength at which the beta coefficients are computed. The default is
            the center of the simulation axis.
        order : int, optional
            Highest order of the beta coefficients to be computed. The default is 6.

        Returns
        -------
        beta : array
            Beta coefficients of the material at the given wavelength.
        """
        if wavelength is None:
            center_omega = self.center_omega
        else:
            center_omega = 2*np.pi*c/wavelength
        
        om = 2*np.pi*self.freq
        dk = np.real(self.n_eff)*om/c
        b_inter = interp1d(om, dk, kind='cubic')
        beta = np.zeros(order+1)
        beta[0] = b_inter(center_omega)
        for i in np.arange(order):
            dk = np.diff(dk)
            dom = np.diff(om)
            om = om[:-1] + dom/2
            der = dk/dom
            der_inter = interp1d(om, der, kind='cubic')
            beta[i+1] = der_inter(center_omega)
            dk = der
        return beta
    
    def set_betas(self, betas, wavelength):
        """
        Convenience only. Allow the use to provide the beta coefficients
        instead of n_eff. Should be avoided.
        """
        rel_om = self.omega - 2*np.pi*c/wavelength
        n = np.ones(self.t_pts)*betas[0]
        factorial = 1
        for i, b in enumerate(betas[1:]):
            factorial *= (i+1)
            n += b*rel_om**(i+1)/factorial
        n *= c/self.omega
        self.n_eff = n

    def set_n2(self, n2, wavelength=None):
        """
        Convenience only. Allow the user to give the nonlinear index instead
        of chi3. Overwrite the initialized chi3 value accordingly.

        Parameters
        ----------
        n2 : float
            Nonlinear index.
        """
        if wavelength is None:
            freq = self.center_freq
        else:
            freq = c/wavelength
        self.chi_3 = 4*n2*eps_0*c*self.n_eff_inter(freq)**2/3
        
    def set_gamma(self, gamma, wavelength=None):
        """
        Convenience only. Allow the user to give the nonlinear coefficient instead
        of chi3. Overwrite the initialized chi3 value accordingly.

        Parameters
        ----------
        gamma : float
            Nonlinear coefficient.
        """
        if wavelength is None:
            freq = self.center_freq
        else:
            freq = c/wavelength
        self.chi_3 = 2*self.n_eff_inter(freq)**2*c**2*eps_0*gamma*self.eff_area/3/np.pi/freq

    def plot_refractive_index(self, savename=None):
        """
        Plot refractive index
        """        
        k = self.n_eff*2*np.pi*self.freq/c
        inverse_group_velocity = np.gradient(k, 2*np.pi*self.freq)
        group_velocity_dispersion = np.gradient(inverse_group_velocity, 2*np.pi*self.freq)
        
        fig, axs = plt.subplots(3)
        axs[0].plot(self.freq, self.n_eff)
        axs[0].set_ylabel('Refractive index')
        axs[1].plot(self.freq, inverse_group_velocity)
        axs[1].set_ylabel('IGV [s/m]')
        axs[2].plot(self.freq, group_velocity_dispersion)
        axs[2].set_ylabel('GVD [s^2/m]')
        axs[2].set_xlabel('Frequency [Hz]')
        fig.tight_layout()
