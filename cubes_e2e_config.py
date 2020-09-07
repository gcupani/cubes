from cubes_etc_parameters import *
from astropy import units as au
from astropy.io import ascii
import numpy as np

# See http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
wave_ref_AB = {'u': 356*au.nm, 
               'g': 483*au.nm, 
               'r': 626*au.nm, 
               'i': 767*au.nm, 
               'z': 910*au.nm}
wave_ref_Vega = {'U': 360*au.nm,
                 'B': 438*au.nm,
                 'V': 545*au.nm,
                 'R': 641*au.nm,
                 'I': 798*au.nm,
                 'J': 1220*au.nm,
                 'H': 1630*au.nm,
                 'K': 2190*au.nm}
                 
                 
flux_ref_AB = {'u': 15393*au.photon/au.cm**2/au.s/au.nm, 
               'g': 11346*au.photon/au.cm**2/au.s/au.nm, 
               'r': 8754*au.photon/au.cm**2/au.s/au.nm, 
               'i': 7145*au.photon/au.cm**2/au.s/au.nm, 
               'z': 6022*au.photon/au.cm**2/au.s/au.nm}
flux_ref_Vega = {'U': 7561*au.photon/au.cm**2/au.s/au.nm,
                 'B': 13926*au.photon/au.cm**2/au.s/au.nm,
                 'V': 9955*au.photon/au.cm**2/au.s/au.nm,
                 'R': 7020*au.photon/au.cm**2/au.s/au.nm,
                 'I': 4520*au.photon/au.cm**2/au.s/au.nm,
                 'J': 1931*au.photon/au.cm**2/au.s/au.nm,
                 'H': 933*au.photon/au.cm**2/au.s/au.nm,
                 'K': 436*au.photon/au.cm**2/au.s/au.nm}
    
wave_U = 360 * au.nm  # Effective wavelength, U band
wave_u = 356 * au.nm  # Effective wavelength, u band
flux_U = 7561 * au.photon / au.cm**2 / au.s / au.nm  # Flux density @ 360.0 nm, mag_U = 0 (Vega)
flux_u = 15393 * au.photon / au.cm**2 / au.s / au.nm  # Flux density @ 356.0 nm, mag_u = 0 (AB)

arm_n = 2  # Number of arms

eff_wave = [300, 305, 310, 320, 325, 330, 340, 350, 360, 370, 375, 380, 385, 390, 400, 410, 420]*au.nm  # Wavelengths used to model efficiency
eff_fib = [0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96]  # Fiber efficiency
eff_adc = [0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94]  # ADC efficiency
eff_opo = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]  # Other pre-optics efficiency
eff_slc = [0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]  # Slicer efficiency
eff_dch = [0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96]  # Dichroics efficiency
eff_fld = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]  # Fold efficiency
eff_col = [0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96]  # Collimator efficiency
eff_cam = [0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]  # Camera efficiency
#sim.eff_spc = [0.79, 0.89, 0.94, 0.94, 0.92, 0.90]  # Spectrograph efficiency
if arm_n == 1:
    eff_grt = [0.79, 0.84, 0.88, 0.91, 0.93, 0.95, 0.97, 0.96, 0.94, 0.91, 0.88, 0.87, 0.85, 0.83, 0.78, 0.72, 0.65]  # Grating efficiency
if arm_n == 2:
    eff_grt = [0.90, 0.91, 0.93, 0.96, 0.97, 0.97, 0.95, 0.92, 0.93, 0.96, 0.97, 0.97, 0.96, 0.95, 0.92, 0.88, 0.85]  # Grating efficiency
if arm_n == 3:
    eff_grt = [0.91, 0.93, 0.95, 0.97, 0.97, 0.95, 0.95, 0.97, 0.96, 0.94, 0.95, 0.96, 0.97, 0.97, 0.95, 0.92, 0.88]  # Grating efficiency
eff_ccd = [0.78, 0.79, 0.80, 0.80, 0.80, 0.80, 0.80, 0.79, 0.77, 0.72, 0.72, 0.72, 0.72, 0.72, 0.74, 0.76, 0.75]  # CCD QE
eff_tel = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]  # Telescope efficiency
resol = [2.0e4, 2.1e4, 2.2e4]  # Instrument resolution

ccd_xsize = 6144*au.pixel  # X size of the CCD
ccd_ysize = 6144*au.pixel  # Y size of the CCD
ccd_xbin = 1  # X binning of the CCD
ccd_ybin = 1  # Y binning of the CCD
pix_xsize = pixsize*au.micron  # X size of the pixels (from cubes_etc_parameters.py)
pix_ysize = pixsize*au.micron  # Y size of the pixels (from cubes_etc_parameters.py)
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
arm_n = 2  # Number of arms
if arm_n == 2:
    wave_d = [347.5]*au.nm  # Wavelength of dichroich
    wave_d_shift = 2*au.nm  # Shift of the dichroich wavelength from the CCD edge
if arm_n == 3:
    wave_d = [335, 362.5]*au.nm  # Wavelength of dichroichswave_d = [336, 367]*au.nm
    wave_d_shift = 4*au.nm  # Shift of the dichroich wavelength from the CCD edge
wave_sampl = [8.1e-3, 8.8e-3, 8.9e-3]*au.nm/au.pixel
disp_wave = [300, 320, 340, 360, 380, 400]*au.nm  # Wavelengths used to model dispersion
disp_sampl = [8.2e-3, 7.6e-3, 7.0e-3, 8.3e-3, 7.7e-3, 7.1e-3]*au.nm/au.pixel  # Dispersion sampling on CCD
disp_resol = [1.86e4, 1.91e4, 1.96e4, 2.01e4, 2.06e4, 2.11e4]  # Resolution
slice_n = 6  # Number of slices
slice_length = 10*au.arcsec  # Length of the slice
slice_width = 0.25*au.arcsec  # Width of the slice
slice_gap = 40*au.pixel  # Length of the slice
ccd_bias = 100*au.adu
#ccd_ron = 2*au.adu
ccd_gain = 1.1*au.photon/au.adu
ccd_ron = ron*au.photon / ccd_gain  # (From cubes_etc_parameters.py)
#ccd_dark = 0.5*au.adu/au.h
ccd_dark = 3*au.photon/au.h / ccd_gain

seeing = 0.87*au.arcsec  # Seeing
psf_func = 'gaussian'  # Function to represent the PSF ('tophat', 'gaussian', 'moffat')
psf_sampl = 1000*au.pixel  # Size of the PSF image
psf_cen = (0,0)  # Center of the PSF
area = (400*au.cm)**2 * np.pi  # Telescope area
texp = 3600*au.s  # Exposure time
mag_syst = 'AB'  # Magnitude system
mag_band = 'r'  # Magnitude band
targ_mag = 17  # Magnitude of the target @ 350 nm
bckg_mag = 22.5  # Magnitude of the background @ 350 nm
airmass = 1.16  # Airmass
pwv = 10.0  # Precipitable water vapor
moond = 0  # Days from new moon

spec_templ = 'qso'  # Function for the template spectrum ('flat', 'PL', 'qso', 'star')
spec_file = 'Zheng+97.txt'
zem = None
igm_abs = None
#star_file = 'Castelliap000T5250g45.dat'
extr_func = 'sum'  # Function for extracting the spectrum ('sum', 'opt' [very slow])
snr_sampl = 1*au.nm  # Data points per SNR point

phot_pars = ['bckg_mag', 'mag_syst', 'mag_band', 'targ_mag', 'texp']
spec_pars = ['spec_templ', 'spec_file', 'qso_zem', 'qso_lya_abs', 'airmass', 'pwv', 'moond']
psf_pars = ['psf_func', 'psf_sampl', 'psf_cen', 'slice_n', 'slice_length', 'slice_width', 'seeing']
ccd_pars = ['ccd_gain', 'ccd_ron', 'ccd_dark', 'ccd_xsize', 'ccd_ysize', 'pix_xsize', 'pix_ysize', 'ccd_xbin', 'ccd_ybin',
            'arm_n', 'wave_d', 'wave_d_shift', 'eff_adc', 'eff_slc', 'eff_dch', 'eff_spc', 'eff_grt', 'eff_ccd', 'disp_wave', 
            'disp_sampl', 'disp_resol', 'spat_scale', 'slice_gap', 'extr_func']