# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:31:47 2022

The waveguide/fiber parameters are first provided, and a Waveguide instance
is created. Then, the pulse parameters are used to create a Light object.
A physical model is then chosen, taking into account different nonlinear
interactions based on the user choice. Finally, a solver is instantiated
and computes the propagation of the pulse in the waveguide with the chosen
nonlinear interactions.

@author: voumardt
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c

import sys
sys.path.append(os.getcwd() + '../../../')

from src import  *


"""
User parameters
"""
### Simulation
t_pts = 2**17

### Light
pulse_duration = 100e-15
pulse_wavelength = 1.56e-06
pulse_energy = 1.6e-9

### Waveguide
wg_length = 0.001
wg_chi_2 = 1.1e-12
wg_chi_3 = 3.4e-21
wg_a_eff = 1e-12
wg_freq, wg_n_eff = np.load('effective_index.npy')


"""
Nonlinear propagation
"""
### Prepare waveguide
waveguide = materials.Waveguide(wg_freq, wg_n_eff, wg_chi_2, wg_chi_3,
                                wg_a_eff, wg_length, t_pts=t_pts)

### Prepare input pulse
pulse = light.Sech(waveguide, pulse_duration, pulse_energy, pulse_wavelength)

### Prepare model
model = models.SpmChi2Chi3(waveguide, pulse)

### Prepare solver, solve
solver = solvers.Solver(model)
solver.solve()


"""
Plots
"""
pulse.plot_propagation('cascaded.png')

