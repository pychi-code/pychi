# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:01:48 2021

@author: Thibault

This module contains the classes used to instantiate light. Different
standard pulse shapes are built-in, and a class is dedicated to accomodating
user-provided light with arbitrary spectral shape. Note that the electric
field is stored as an analytical envelope, thus is a complex quantity.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, h
from scipy.interpolate import interp1d
eps_0 = 8.8541878128e-12


class Light():
    """
    Parent light class. Implement support for adding pulses,
    saving the results of a propagation and plotting these
    results. Not suitable for light instantiation.
    """
    def __init__(self, waveguide, field_t_in):
        """
        Light is here defined by a time axis coming from a
        waveguide object, and a temporal waveform.

        Parameters
        ----------
        waveguide : Waveguide
            Waveguide object containing the material properties.
        field_t_in : array
            Array containing the complex temporal waveform of the
            initial analytical electric field.
        """
        self.waveguide = waveguide
        self.field_t_in = field_t_in
        self.eff_area_pump = self._compute_eff_area_pump()
    
    def __add__(self, light):
        """
        Addition operator overload. Allow one to add pulses easily.

        Parameters
        ----------
        light : Light object
            Light with field to be added.

        Returns
        -------
        Light
            New Light object with field added.
        """
        tot_field = self.field_t_in + light.field_t_in
        return Light(self.waveguide, tot_field)
        
    def _set_propagation(self, z_save, freq, spectrum, waveform):
        """
        Store propagation results. Called from the solver module
        after finishing propagation.
        """
        self.z_save = z_save
        self.freq = freq
        self.spectrum = spectrum
        self.waveform = waveform
        self.wl, self.spectrum_wl = self._compute_equidistant_wl_spectrum()
        
    def _compute_equidistant_wl_spectrum(self):
        """
        Compute a properly normalized field representation in the wavelength
        domain with equidistantly spaced sampling points.

        Returns
        -------
        array
            Field in the wavelength domain.
        """
        wl = c/self.freq
        wl_ener = np.abs(self.freq*self.spectrum/np.sqrt(c))**2*wl
        wl_ener_interp = interp1d(wl, wl_ener, kind='linear')
        wl_eq = np.linspace(np.amin(wl), np.amax(wl), len(self.field_t_in))
        ener_wl_eq = wl_ener_interp(wl_eq)
        return wl_eq, np.sqrt(ener_wl_eq/wl_eq)
        
    def _set_dB(self):
        """
        Return a function to compute the dB representation of a field
        normalized to the input light spectrum.
        """
        if self.spectrum is not None:
            return lambda field: 20*np.log10(np.abs(field) + 1e-20) - np.amax(20*np.log10(np.abs(self.spectrum[0]) + 1e-20))
        else:
            pass

    def _compute_eff_area_pump(self):
        """
        Compute the effective area at z=0 at the pump wavelength
        """
        if len(np.shape(self.waveguide.eff_area)) == 0:
            return self.waveguide.eff_area
        elif len(self.waveguide.eff_area) == len(self.waveguide.freq):
            eff_area_inter = interp1d(self.waveguide.freq, self.waveguide.eff_area, kind='cubic',
                                    bounds_error=False, fill_value='extrapolate')
            return eff_area_inter(c/self.pulse_wavelength)
        else:
            raise ValueError('Invalid effective area format!')

    def add_group_delay_dispersion(self, d2):
        """
        Add group delay dispersion (chirp) to the light field.

        Parameters
        ----------
        d2 : float
            Group delay dispersion
        """
        field_angle = np.angle(self.field_t_in)
        field_angle *= d2/2*self.waveguide.rel_omega**2
        self.field_t_in = np.abs(self.field_t_in)*np.exp(1j*field_angle)
        
    def add_shot_noise(self, seed=None):
        """
        Add shot noise to the pulse, useful for coherence study.

        Parameters
        ----------
        seed : int
            Seed for the random number generator for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            
        field_angle = np.angle(self.field_t_in)
        
        e_to_n_fac = eps_0*self.waveguide.n_eff_inter(c/self.pulse_wavelength)
        e_to_n_fac *= self.pulse_wavelength*self.eff_area_pump
        e_to_n_fac *= self.waveguide.delta_t/2/h
        
        photons_per_bin = e_to_n_fac*np.abs(self.field_t_in)**2
        photons_per_bin = np.random.poisson(photons_per_bin)

        noisy_field = np.sqrt(photons_per_bin/e_to_n_fac).astype(np.complex128)
        noisy_field *= np.exp(1j*field_angle)
        self.field_t_in = noisy_field
        
    def plot_propagation(self, savename=None):
        """
        Plot propagation results.
        """
        if self.spectrum is not None:
            self.dB = self._set_dB()
            ### Overall figure
            fig = plt.figure(figsize=(10,10))
            ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
            ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
            ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
            ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)
            
            # Input and output spectrum
            ax0.plot(self.freq, self.dB(self.spectrum)[0], color='b')
            ax0.plot(self.freq, self.dB(self.spectrum)[-1], color='r')
            ax0.set_ylabel('Power density [dB]')
            ax0.set_ylim(-100, 5)
            
            # Input and output pulse
            ax1.plot(self.waveguide.time, np.abs(self.waveform)[-1], color='r')
            ax1.plot(self.waveguide.time, np.abs(self.waveform)[0], color='b')
            
            # Spectrum color map
            extent = (np.amin(self.freq[self.freq>0]), np.max(self.freq), 0, self.waveguide.length)
            ax2.imshow(self.dB(self.spectrum[:, self.freq>0]), extent=extent, vmin=np.amax(self.dB(self.spectrum)) - 100.0,
                              vmax=np.amax(self.dB(self.spectrum)), aspect='auto', origin='lower')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('Propagation distance [m]')
            
            # Pulse color map
            extent = (np.min(self.waveguide.time), np.max(self.waveguide.time), 0, self.waveguide.length)
            plt_waveform = np.abs(self.waveform)
            ax3.imshow(plt_waveform, extent=extent, vmin=np.amin(plt_waveform[-1]),
                        vmax=np.amax(plt_waveform), aspect='auto', origin='lower')
            ax3.set_xlabel('Time [s]')
            
            plt.suptitle('Propagation results')
            fig.tight_layout()
            plt.show()
            if savename is not None:
                plt.savefig(savename)

            ### Spectrogram
            plt.figure(22222)
            specgram, _, _, _ = plt.specgram(self.waveform[-1],
                         Fs=np.amax(self.freq)-np.amin(self.freq),
                         Fc=(np.amax(self.freq)+np.amin(self.freq))/2,
                         xextent=(np.amin(self.waveguide.time), np.amax(self.waveguide.time)),
                         mode='magnitude',
                         scale='dB',
                         sides='twosided')
            plt.close(22222)
            plt.figure()
            ax0 = plt.subplot(221)
            plt.specgram(self.waveform[-1],
                         Fs=np.amax(self.freq)-np.amin(self.freq),
                         Fc=(np.amax(self.freq)+np.amin(self.freq))/2,
                         xextent=(np.amin(self.waveguide.time), np.amax(self.waveguide.time)),
                         mode='magnitude',
                         scale='dB',
                         vmax=20*np.log10(np.amax(specgram)),
                         vmin=20*np.log10(np.amax(specgram))-100,
                         sides='twosided')
            plt.ylabel('Frequency [Hz]')
            
            plt.subplot(222, sharey=ax0)
            plt.plot(20*np.log10(np.abs(self.spectrum[-1])) - np.amax(20*np.log10(np.abs(self.spectrum[-1]))), self.freq)
            plt.xlabel('Intensity [dB]')
            plt.xlim(5, -100)
            plt.ylim((np.amin(self.freq), np.amax(self.freq)))
            
            plt.subplot(223, sharex=ax0)
            plt.plot(self.waveguide.time, np.abs(self.waveform[-1])**2)
            plt.xlabel('Time [s]')
            plt.ylabel('Intensity [a.u.]')
            plt.xlim((np.amin(self.waveguide.time), np.amax(self.waveguide.time)))
            plt.tight_layout()
            if savename is not None:
                plt.savefig('spec_' + savename)
            
        else:
            pass
        
    def set_result_as_start(self, waveguide=None):
        """
        Set output field from propagation as initial field for a new
        propagation.

        Returns
        -------
        Light
            New Light object with final propagation result as input field.
        """
        if waveguide is None:
            waveguide = self.waveguide
        if self.spectrum is not None:
            field_t_in = self.waveform[-1]
            return Light(waveguide, field_t_in)
        else:
            pass


class Sech(Light):
    """
    Class for a sech pulse.
    """
    def __init__(self, waveguide, pulse_duration, pulse_energy,
                 pulse_wavelength, delay=0):
        """
        Construct a sech pulse.

        Parameters
        ----------
        waveguide : Waveguide
            Waveguide object containing the material properties.
        pulse_duration : float
            Pulse FWHM duration.
        pulse_energy : float
            Energy per pulse.
        pulse_wavelength : float
            Central wavelength of the pulse.
        delay: float, defaults to 0.
            Time offset of the pulse.
        """
        self.pulse_duration = pulse_duration
        self.pulse_energy = pulse_energy
        self.pulse_wavelength = pulse_wavelength
        self.waveguide = waveguide
        self.delay = delay
        self.eff_area_pump = self._compute_eff_area_pump()
        self.field_t_in = self._compute_field()
    
    def _compute_field(self):
        """
        Instantiate electric field as a sech pulse.

        Returns
        -------
        array
            Analytical envelope of the electric field.
        """
        pulse_peak_power = 0.8813736*self.pulse_energy/self.pulse_duration
        field_t_in = np.sqrt(pulse_peak_power)/np.cosh(1.7627472*(self.waveguide.time - self.delay)/self.pulse_duration)\
            *np.exp(1j*(2*np.pi*c/self.pulse_wavelength - self.waveguide.center_omega)*(self.waveguide.time - self.delay))
        field_t_in /= np.sqrt(self.waveguide.n_eff_inter(c/self.pulse_wavelength)*eps_0*c*self.eff_area_pump/2)
        return field_t_in


class Gaussian(Light):
    """
    Class for a Gaussian pulse.
    """
    def __init__(self, waveguide, pulse_duration, pulse_energy,
                 pulse_wavelength, delay=0):
        """
        Construct a Gaussian pulse.

        Parameters
        ----------
        waveguide : Waveguide
            Waveguide object containing the material properties.
        pulse_duration : float
            Pulse FWHM duration.
        pulse_energy : float
            Energy per pulse.
        pulse_wavelength : float
            Central wavelength of the pulse.
        delay: float, defaults to 0.
            Time offset of the pulse.
        """
        self.pulse_duration = pulse_duration
        self.pulse_energy = pulse_energy
        self.pulse_wavelength = pulse_wavelength
        self.waveguide = waveguide
        self.delay = delay
        self.eff_area_pump = self._compute_eff_area_pump()
        self.field_t_in = self._compute_field()
    
    def _compute_field(self):
        """
        Instantiate electric field as a gaussian pulse.

        Returns
        -------
        array
            Analytical envelope of the electric field.
        """
        amplitude = 2*self.pulse_energy*np.sqrt(8*np.log(2)/np.pi)
        amplitude /= self.waveguide.n_eff_inter(c/self.pulse_wavelength)
        amplitude /= eps_0*c*self.pulse_duration*self.eff_area_pump
        field_t_in = np.sqrt(amplitude)*np.exp(-4*np.log(2)*(self.waveguide.time - self.delay)**2/self.pulse_duration**2)\
            *np.exp(1j*(2*np.pi*c/self.pulse_wavelength - self.waveguide.center_omega)*(self.waveguide.time - self.delay))
        return field_t_in


class Cw(Light):
    """
    Class for CW light.
    """
    def __init__(self, waveguide, average_power, wavelength):
        """
        Construct CW light.
        
        Parameters
        ----------
        waveguide : Waveguide
            Waveguide object containing the material properties.
        pulse_average_power : float
            Average power of the CW light.
        pulse_wavelength : float
            Central wavelength of the light.
        """
        self.pulse_average_power = average_power
        self.pulse_wavelength = wavelength
        self.waveguide = waveguide
        self.eff_area_pump = self._compute_eff_area_pump()
        self.field_t_in = self._compute_field()
        
    def _compute_field(self):
        """
        Instantiate electric field as a continuous wave.

        Returns
        -------
        array
            Analytical envelope of the electric field.
        """
        field_t_in = np.exp(1j*(2*np.pi*c/self.pulse_wavelength - self.waveguide.center_omega)*self.waveguide.time)
        field_t_in *= np.sqrt(2*self.pulse_average_power/self.waveguide.n_eff_inter(c/self.pulse_wavelength)/eps_0/c/self.eff_area_pump)
        return field_t_in
    

class Arbitrary(Light):
    """
    Class for arbitrary light waveform, specified in frequency domain.
    """
    def __init__(self, waveguide, pulse_frequency_axis, pulse_electric_field,
                 pulse_energy):
        """
        Construct an arbitrary waveform from its frequency domain field.

        Parameters
        ----------
        waveguide : Waveguide
            Waveguide object containing the material properties.
        pulse_frequency_axis : array
            Frequency axis associated to the field.
        pulse_electric_field : array
            Analytical envelope representation of electric field in the frequency domain.
        pulse_energy : float
            Energy per pulse, used to normalize the field amplitude.
        """
        self.pulse_energy = pulse_energy
        self.data_freq = pulse_frequency_axis
        self.data_field = pulse_electric_field
        self.pulse_wavelength = self._compute_center_wavelength()
        self.waveguide = waveguide
        self.eff_area_pump = self._compute_eff_area_pump()
        self.field_t_in = self._compute_field()
    
    def _compute_center_wavelength(self):
        """
        Compute the carrier wavelength of the pulse.
        """
        pw = np.sum(np.abs(self.data_field)*self.data_freq)
        pw /= np.sum(np.abs(self.data_field))
        return c/pw
        
    def _compute_field(self):
        """
        Instantiate input light as a user-specified waveform.

        Returns
        -------
        array
            Analytical envelope of the electric field.
        """
        field_interp = interp1d(self.data_freq, self.data_field, kind='cubic',
                                bounds_error=False, fill_value=0.0)
        field_f_in = field_interp(self.waveguide.freq)
        field_t_in = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(field_f_in)))
        temp = np.sum(np.abs(field_t_in)**2)*self.waveguide.delta_t
        field_t_in *= np.sqrt(2*self.pulse_energy/(temp*self.waveguide.n_eff_inter(c/self.pulse_wavelength)*eps_0*c*self.eff_area_pump))
        return field_t_in
        
