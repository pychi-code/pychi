# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:31:47 2022

@author: voumardt
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
eps_0 = 8.8541878128e-12
import math
import os

import sys
sys.path.append(os.getcwd() + '../../../')

from src import  *


"""
User parameters

Parameters taken from 'Supercontinuum generation in photonic crystal fiber',
John M. Dudley, Goëry Genty, and Stéphane Coen, DOI: https://doi.org/10.1103/RevModPhys.78.1135
to reproduce figure 3 therein.
"""
### Simulation
t_pts = 2**14

### Light
pulse_duration = 50e-15
pulse_wavelength = 0.835e-06
pulse_peak_power = 1e4
pulse_energy = pulse_peak_power*pulse_duration/np.log(1+2**0.5)

### Waveguide
wg_length = 0.15
wg_chi_2 = 0
wg_a_eff = 1e-12
wg_n = 1.45
gamma = 0.11
wg_chi_3 = 4*wg_n**2*c*eps_0*gamma*pulse_wavelength*wg_a_eff/6/np.pi
wg_omega = np.linspace(c/1670e-9*2*np.pi, c/417.5e-9*2*np.pi, t_pts)
betas = [wg_n*2*np.pi/pulse_wavelength, wg_n/c, -1.183e-26, 8.1038e-41,
         -9.5205e-56, 2.0737e-70, -5.3943e-85, 1.3486e-99, -2.5495e-114,
         3.0524e-129, -1.7140e-144]

### Compute refractive index
wg_freq = wg_omega/2/np.pi
def compute_n_eff(wg_omega, pulse_wavelength, betas):
    k = np.zeros(t_pts, dtype='float64')
    for i, beta in enumerate(betas):
        k += beta*(wg_omega - 2*np.pi*c/pulse_wavelength)**i/math.factorial(i)
    n_eff = k*c/wg_omega
    return n_eff
wg_n_eff = compute_n_eff(wg_omega, pulse_wavelength, betas)


"""
Nonlinear propagation
"""
### Prepare waveguide
waveguide = materials.Waveguide(wg_freq, wg_n_eff, wg_chi_2, wg_chi_3,
                                wg_a_eff, wg_length, t_pts=t_pts)

### Prepare input pulse
pulse = light.Sech(waveguide, pulse_duration, pulse_energy, pulse_wavelength)

### Prepare model
model = models.Spm(waveguide, pulse)

### Prepare solver
solver = solvers.Solver(model)

### Solve
solver.solve()


"""
Plots
"""
z_pos = pulse.z_save
wl = pulse.wl
spec_wl = pulse.spectrum_wl[:, (wl>450e-9)&(wl<1300e-9)]
wl = wl[(wl>450e-9)&(wl<1300e-9)]
spec_wl_db = 20*np.log10(np.abs(spec_wl) + 1e-20)


plt.figure()
plt.subplot(121)
plt.imshow(spec_wl_db[::-1], cmap='jet', aspect='auto', vmin=np.amax(spec_wl_db) - 40,
           extent=(np.amin(wl)*1e9, np.amax(wl)*1e9, 0, np.amax(z_pos)))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Distance [m]')

waveform = 10*np.log10(np.abs(pulse.waveform)**2 + 1e-20) - np.amax(10*np.log10(np.abs(pulse.waveform)**2 + 1e-20))
time = waveguide.time
waveform = waveform[:, (time<5e-12)&(time>-2e-12)]
time = time[(time<5e-12)&(time>-2e-12)]

plt.subplot(122)
plt.imshow(waveform[::-1], cmap='jet', aspect='auto', vmin = np.amax(waveform) - 40,
           extent=(np.amin(time)*1e12, np.amax(time)*1e12, 0, np.amax(z_pos)))
plt.xlabel('Time [ps]')
plt.ylabel('Distance [m]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.savefig('propagation.png')
