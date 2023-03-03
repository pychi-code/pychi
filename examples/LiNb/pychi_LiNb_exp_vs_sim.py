# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:31:47 2022

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
t_pts = 2**16

### Light
pulse_duration = 80e-15
pulse_wavelength = 1.58e-06
pulse_energy = 1.056e-11

### Waveguide
wg_length = 0.006
wg_chi_2 = 20e-12
wg_chi_3 = 2.5e-21 # Raman lasing and soliton mode-locking in lithium niobate microresonators
wg_a_eff = 1e-12
wg_freq, wg_n_eff = np.load('n_eff_data_LiNb.npy')


"""
Nonlinear propagation
"""
waveguide = materials.Waveguide(wg_freq, wg_n_eff, wg_chi_2, wg_chi_3,
                                wg_a_eff, wg_length, t_pts=t_pts)
pulse = light.Sech(waveguide, pulse_duration, pulse_energy, pulse_wavelength)
model = models.SpmChi2(waveguide, pulse)
solver = solvers.Solver(model)
solver.solve()


"""
Plots
"""
pulse.plot_propagation()

# Load experimental results
exp_data = loadmat('exp_LiNb.mat')
exp_wl = exp_data['wavelength'][0]
exp_int = exp_data['intensity'][0]

# Compare experimental results and simulation
plt.figure()
plt.plot(pulse.wl, 10*np.log10(pulse.spectrum_wl[-1]/np.amax(pulse.spectrum_wl[-1]))-3)
plt.plot(exp_wl, exp_int - np.amax(exp_int))
plt.xlim(3.5e-7, 1.75e-6)
plt.ylim(-75, 5)
plt.title('Experimental vs simulation - LiNb')
plt.xlabel('Wavelength [m]')
plt.ylabel('Intensity [dB]')
plt.legend(('Simulation', 'Experiment'))
plt.savefig('Experimental_vs_simulation_LiNb.png')
