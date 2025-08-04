# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:54:20 2022

@author: voumardt
"""
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.constants import c
import time as timing
eps_0 = 8.8541878128e-12


class Solver():
    def __init__(self, model, z_pts=500, local_error=0.00001, adaptive_factor=1.1,
                 max_dz=None, method=None, breakpoints=[]):
        """
        Solver class. Provides a stepper which performs integration of the PDE
        as well as error control and results saving.

        Parameters
        ----------
        model : Model object
            Physical model of the problem to be integrated.
        z_pts : int, optional
            Number of results to keep along the propagation axis. The default is 500.
        local_error : float, optional
            Tolerated local error. The default is 0.00001.
        adaptive_factor : float, optional
            Adaptive factor for the step size control. Bigger values lead to
            more aggressive step size control. The default is 1.1.
        max_dz : float, optional
            Maximum step size allowed. Particularly useful when simulating
            quasi-phase matched structures. The default is None.
        method: string
            Method used for integration. Can be RK4IP, ERK4IP or DP5IP.
            Default is DP5IP.
        breakpoints: array
            This array contains z coordinates where the solver will be forced
            to compute a result. Useful for poled structures, to avoid having
            a step larger than the poling. The default is an empty array.
        """
        self.model = model
        self.z_pts = z_pts
        self.local_error = local_error
        self.adaptive_factor = adaptive_factor
        self.max_dz = max_dz
        self.z_save = np.linspace(0, self.model.waveguide.length, self.z_pts)
        self.method_name = self._set_method_name(method)
        self.breakpoints = breakpoints
        self.stepsize_save = []
        self.power_prefactor = eps_0*c/2/self.model.waveguide.t_pts*self.model.waveguide.delta_t
    
    def _set_method_name(self, method):
        """
        Helper function to choose an appropriate solver. If a method is not
        provided by the user, it  will look into the model used to see if
        there is a recommended solver. Otherwise, defaults to the model that
        should be mostly used, i.e. DP5IP.

        Parameters
        ----------
        method : string
            Name of the solver.

        Returns
        -------
        string
            Name of the solver.
        """
        if method is None:
            if hasattr(self.model, '_recommended_solver'):
                return self.model._recommended_solver
            else:
                return 'DP5IP'
        else:
            return method
        
    def _compute_error(self, field_1, field_2):
        """
        Error estimation for adaptive step size

        Parameters
        ----------
        field_1 : array
            Reference field.
        field_2 : array
            Test field.

        Returns
        -------
        float
            Error estimate between the two fields.
        """
        return numbanorm(field_1 - field_2)/numbanorm(field_1)
    
    def _dormand_prince_step(self, field, dz):
        """
        Dormand Prince order 5(4) adaptive scheme in the interaction picture. This
        new stepper uses the interaction picture in an embedded scheme of order 5.
        Tests have shown it to be up to four times faster than RK4IP.

        Parameters
        ----------
        field : array
            Current field.
        dz : float
            Current step size.

        Returns
        -------
        array
            New field estimate.
        
        float
            Error estimate.
        """
        if self._k_7 is None:
            k_1 = dz*self.model.dispersion_step(self.model.nonlinear_term(field), dz)
        else:
            k_1 = self.model.dispersion_step(self._k_7, dz)
        a_int = self.model.dispersion_step(field, dz)
        k_2 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(a_int + k_1/5, -4*dz/5)), 4*dz/5)
        k_3 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(a_int + 3*k_1/40 + 9*k_2/40, -7*dz/10)), 7*dz/10)
        k_4 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(a_int + 44*k_1/45 - 56*k_2/15 + 32*k_3/9, -dz/5)), dz/5)
        k_5 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(a_int + 19372*k_1/6561 - 25360*k_2/2187 + 64448*k_3/6561 - 212*k_4/729, -dz/9)), dz/9)
        k_6 = dz*self.model.nonlinear_term(a_int + 9017*k_1/3168 - 355*k_2/33 + 46732*k_3/5247 + 49*k_4/176 - 5103*k_5/18656)
        sol_5 = a_int + 35*k_1/384 + 500*k_3/1113 + 125*k_4/192 - 2187*k_5/6784 + 11*k_6/84
        self._k_7 = dz*self.model.nonlinear_term(sol_5)
        sol_4 = a_int + 5179*k_1/57600 + 7571*k_3/16695 + 393*k_4/640 - 92097*k_5/339200 + 187*k_6/2100 + self._k_7/40
        error = self._compute_error(sol_5, sol_4)
        return sol_5, error
    
    def _rk546m(self, field, dz):
        """
        Some other Dormand-Prince method that has one less step but does not
        leverage the interaction picture and does not have the FSAL property.
        Usually worse than the other dormand-prince method.
        References:
            DOI: 10.1016/0771-050X(80)90013-3

        Parameters
        ----------
        field : array
            Current field.
        dz : float
            Current step size.

        Returns
        -------
        array
            New field estimate.
        
        float
            Error estimate.
        """
        k_1 = dz*self.model.nonlinear_term(field)
        k_2 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(field + k_1/5, dz/5)), -dz/5)
        k_3 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(field + 3*k_1/40 + 9*k_2/40, 3*dz/10)), -3*dz/10)
        k_4 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(field + 3*k_1/10 - 9*k_2/10 + 6*k_3/5, 3*dz/5)), -3*dz/5)
        k_5 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(field + 226*k_1/729 - 25*k_2/27 + 880*k_3/729 + 55*k_4/729, 2*dz/3)), -2*dz/3)
        k_6 = dz*self.model.dispersion_step(self.model.nonlinear_term(self.model.dispersion_step(field  - 181*k_1/270 + 5*k_2/2 - 266*k_3/297 - 91*k_4/27 + 189*k_5/55, dz)), -dz)
        sol_5 = self.model.dispersion_step(field + 19*k_1/216 + 1000*k_3/2079 - 125*k_4/216 + 81*k_5/88 + 5*k_6/56, dz)
        sol_4 = self.model.dispersion_step(field + 31*k_1/540 + 190*k_3/297 - 145*k_4/108 + 351*k_5/220 + k_6/20, dz)
        error = self._compute_error(sol_5, sol_4)
        return sol_5, error
    
    def _adams_bashforth54_step(self, field, dz):
        """To be implemented"""
        pass

    def _rk4ip_single_step(self, field, dz):
        """
        Runge-Kutta order 4 step in the interaction picture (RK4IP)

        Parameters
        ----------
        field : array
            Current field.
        dz : float
            Current step size.

        Returns
        -------
        array
            New field estimate.
        """
        dispersion_op = numbaexp(-1j*self.model.waveguide.k*dz/2)
        a_int = dispersion_op*field
        k_1 = dz*dispersion_op*self.model.nonlinear_term(field)
        k_2 = dz*self.model.nonlinear_term(a_int + k_1/2)
        k_3 = dz*self.model.nonlinear_term(a_int + k_2/2)
        k_4 = dz*self.model.nonlinear_term(dispersion_op*(a_int + k_3))
        return dispersion_op*(a_int + k_1/6 + k_2/3 + k_3/3) + k_4/6
    
    def _rk4ip_step(self, field, dz):
        """
        Runge-Kutta order 4 adaptive scheme in the interaction picture (RK4IP)
        with step doubling and combining. Provided for comparison but slower
        than the dormand-prince method.
        References:
            DOI: 10.1109/JLT.2007.909373
            DOI: 10.1109/JLT.2009.2021538

        Parameters
        ----------
        field : array
            Current field.
        dz : float
            Current step size.

        Returns
        -------
        array
            New field estimate.
        
        float
            Error estimate.
        """
        sol_single = self._rk4ip_single_step(field, dz)
        sol_double = self._rk4ip_single_step(field, dz/2)
        sol_double = self._rk4ip_single_step(sol_double, dz/2)
        error = self._compute_error(sol_double, sol_single)
        sol_5 = (sol_double*16 - sol_single)/15
        return sol_5, error
    
    def _erk4ip_step(self, field, dz):
        """
        Embedded Runge-Kutta order 4(3) adaptive scheme in the interaction
        picture (ERK4IP). Can be faster than DP5IP if accuracy is irrelevant,
        for example in SPM only cases. Otherwise, DP5IP performs better.
        References:
            DOI: 10.1016/j.cpc.2012.12.020

        Parameters
        ----------
        field : array
            Current field.
        dz : float
            Current step size.

        Returns
        -------
        array
            New field estimate.
        
        float
            Error estimate.
        """
        dispersion_op = numbaexp(-1j*self.model.waveguide.k*dz/2)
        a_int = dispersion_op*field
        if self._k_5 is None:
            k_1 = dz*dispersion_op*self.model.nonlinear_term(field)
        else:
            k_1 = dispersion_op*self._k_5
        k_2 = dz*self.model.nonlinear_term(a_int + k_1/2)
        k_3 = dz*self.model.nonlinear_term(a_int + k_2/2)
        k_4 = dz*self.model.nonlinear_term(dispersion_op*(a_int + k_3))
        beta = dispersion_op*(a_int + k_1/6 + k_2/3 + k_3/3)
        sol_4 = beta + k_4/6
        self._k_5 = dz*self.model.nonlinear_term(sol_4)
        sol_3 = beta + k_4/15 + self._k_5/10
        error = self._compute_error(sol_4, sol_3)
        return sol_4, error
    
    def _integrate_to_z(self, z):
        """
        Integrate to z with adaptive step size.

        Parameters
        -------
        z : float
            Target propagation length.
        """
        while self.stock_z < z:
            # Actuate position inside the waveguide
            self.model.waveguide.z = self.stock_z
            
            # Check for forcing breakpoints
            if (len(self.breakpoints) != 0 and (self.stock_z + self.dz > self.breakpoints[0])):
                
                # Step to breakpoint
                dz = self.breakpoints[0] - self.stock_z + 1e-20
                vec_eval, _ = self.step(self.stock_vec, dz)
                
                # Remove breakpoint
                self.breakpoints = self.breakpoints[1:]
                
                # Update z pos and field
                self.stock_z += dz
                self.stock_vec = vec_eval
                self.stepsize_save = np.append(self.stepsize_save, dz)
            
            else:
            
                # Step of dz
                vec_eval, error = self.step(self.stock_vec, self.dz)
                
                # Error control
                if error >= 2*self.local_error:
                    self.dz /= 2
                    print('Discarded')
                    if self.method_name == 'DP5IP':
                        self._k_7 = None
                    if self.method_name == 'ERK4IP':
                        self._k_5 = None
                    continue
                elif self.local_error <= error < 2*self.local_error:
                    self.stock_z += self.dz
                    self.stepsize_save = np.append(self.stepsize_save, self.dz)
                    self.dz /= self.adaptive_factor
                elif 0.5*self.local_error <= error < self.local_error:
                    self.stock_z += self.dz
                    self.stepsize_save = np.append(self.stepsize_save, self.dz)
                else:
                    self.stock_z += self.dz
                    self.stepsize_save = np.append(self.stepsize_save, self.dz)
                    if self.max_dz is None or self.dz*self.adaptive_factor < self.max_dz:
                        self.dz *= self.adaptive_factor
                self.stock_vec = vec_eval
    
    def solve(self):
        """
        Integrate over waveguide length with adaptive step size
        """
        # Initialization
        self.spectrum = np.zeros((self.z_pts, self.model.waveguide.t_pts), dtype='complex128')
        self.dz = np.mean(np.diff(self.z_save))*np.sqrt(np.sqrt(self.local_error))
        self.stock_z = 0
        self.stock_vec = np.fft.fft(self.model.light.field_t_in)
        if self.method_name == 'DP5IP':
            self._k_7 = None
            self.step = self._dormand_prince_step
        elif self.method_name == 'ERK4IP':
            self._k_5 = None
            self.step = self._erk4ip_step
        elif self.method_name == 'RK4IP':
            self.step = self._rk4ip_step
        elif self.method_name == 'RK546M':
            self.step = self._rk546m
        timer = timing.perf_counter()
            
        # Total integration
        for i, z in enumerate(self.z_save):
            print('z: {}/{}'.format(i+1, self.z_pts))
            
            # Integrate over dz
            self._integrate_to_z(z)
            
            # Save spectrum
            self.spectrum[i, :] = np.fft.fftshift(self.stock_vec)\
                *numbaexp(1j*(self.model.waveguide.betas[0] + self.model.waveguide.betas[1]*self.model.waveguide.rel_omega)*self.stock_z)\
                *np.sqrt(self.model.waveguide.eff_area*self.model.waveguide.n_eff*self.power_prefactor)
        
        # Save waveform
        print(f"{timing.perf_counter() - timer} seconds for integration.")                
        self.waveform = np.fft.ifft(np.fft.ifftshift(self.spectrum, axes=1), axis=1)
        self.model.light._set_propagation(self.z_save, self.model.waveguide.freq, self.spectrum, self.waveform)

    def plot_stepsize(self):
        """
        Plot the evolution of the stepsize against the position in the waveguide
        """
        plt.figure()
        plt.plot(np.cumsum(self.stepsize_save), self.stepsize_save)
        plt.xlabel('Distance [m]')
        plt.ylabel('Step size [m]')
        

"""
Optimized functions
"""
### Numba exponential
@numba.njit([numba.complex128[:](numba.complex128[:]),numba.complex64[:](numba.complex64[:])])
def numbaexp(x):
    return np.exp(x)

### Numba norm
@numba.njit([numba.float64(numba.complex128[:])])
def numbanorm(field):
    return np.linalg.norm(field)
