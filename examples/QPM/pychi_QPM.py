# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:31:47 2022

The waveguide/fiber parameters are first provided, and a Waveguide instance
is created. Then, the pulse parameters are used to create a Light object.
A physical model is then chosen, taking into account different nonlinear
interactions based on the user choice. Finally, a solver is instantiated
and computes the propagation of the pulse in the waveguide with the chosen
nonlinear interactions. This particular examples shows how to define a poling
and breakpoints forcing computation at some desired distances.

@author: voumardt
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
from scipy.constants import c
from scipy.io import loadmat

import sys
sys.path.append(os.getcwd() + '../../../')

from src import  *


"""
User parameters
"""
### Simulation
t_pts = 2**15

### Light
pulse_duration = 100e-15
pulse_wavelength = 1.56e-06
pulse_energy = 1e-9

### Waveguide
wg_length = 0.001
const_wg_chi_2 = 1.1e-12

# poling period
qpm_length = 3.3e-5/2

wg_chi_3 = 3.4e-21
wg_a_eff = 1e-12
wg_freq, wg_n_eff = np.load('effective_index.npy')

# Define the nonlinearity as a function of z
def wg_chi_2(z):
    return (-1)**(np.floor(z/qpm_length)%2)*const_wg_chi_2

# Define breakpoints to force computation whenever the poling changes sign
breakpoints = np.arange(wg_length//qpm_length)*qpm_length

"""
Nonlinear propagation
"""
waveguide = materials.Waveguide(wg_freq, wg_n_eff, wg_chi_2, wg_chi_3,
                                wg_a_eff, wg_length, t_pts=t_pts)
pulse = light.Sech(waveguide, pulse_duration, pulse_energy, pulse_wavelength)
model = models.Chi2(waveguide, pulse)
solver = solvers.Solver(model, breakpoints=breakpoints)
solver.solve()


"""
Plots
"""
pulse.plot_propagation()
