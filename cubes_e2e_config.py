from astropy import units as au
from astropy.io import ascii
import json
import numpy as np

with open('database/request.json') as f:
  jsp = json.load(f)


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

def eff_read(file, col):
    return np.array(ascii.read(jsp['advanced'][file])[col])

eff_file = 'database/EFFICIENCIES'
disp_file = 'database/DISPERSION'

ccd_xsize = int(jsp['advanced']['NXpix'])*au.pixel  # X size of the CCD
ccd_ysize = int(jsp['advanced']['NYpix'])*au.pixel  # Y size of the CCD
ccd_xbin = 1  # X binning of the CCD
ccd_ybin = 1  # Y binning of the CCD
pix_xsize = int(jsp['advanced']['pixsize'])*au.micron  # X size of the pixels
pix_ysize = int(jsp['advanced']['pixsize'])*au.micron  # Y size of the pixels
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
arm_n = 2  # Number of arms
if arm_n == 1:
    wave_d = []*au.nm 
if arm_n == 2:
    wave_d = [349]*au.nm  # Wavelength of dichroich
    wave_d_shift = 2*au.nm  # Shift of the dichroich wavelength from the CCD edge
if arm_n == 3 and ccd_ysize.value < 5000:
    wave_d = [334, 364]*au.nm  # Wavelength of dichroichswave_d = [336, 367]*au.nm
    wave_d_shift = 4*au.nm  # Shift of the dichroich wavelength from the CCD edge
if arm_n == 3 and ccd_ysize.value > 5000:
    wave_d = [340, 380]*au.nm  # Wavelength of dichroichswave_d = [336, 367]*au.nm
    wave_d_shift = 4*au.nm  # Shift of the dichroich wavelength from the CCD edge


slice_n = int(jsp['advanced']['n_slices'])  # Number of slices
slice_length = float(jsp['advanced']['slice_height'])*au.arcsec  # Length of the slice
slice_width = float(jsp['advanced']['slice_width'])*au.arcsec  # Width of the slice
slice_gap = 8*au.pixel  # Length of the slice
ccd_bias = 100*au.adu
ccd_gain = 1.1*au.photon/au.adu
ccd_ron = float(jsp['advanced']['ron'])*au.photon / ccd_gain
ccd_dark = 3*au.photon/au.h / ccd_gain

seeing = float(jsp['seeing'])*au.arcsec  # Seeing
psf_func = 'gaussian'  # Function to represent the PSF ('tophat', 'gaussian', 'moffat')
psf_sampl = 1000*au.pixel  # Size of the PSF image
psf_cen = (0,0)  # Center of the PSF
targ_prof = None
targ_invrad_params = {'r_eff': 20}
targ_sersic_params = {'amplitude': 1, 'r_eff': 20, 'n': 4, 'x_0': 0, 'y_0': 0, 'ellip': 0.5, 'theta': 1}
area = (400*au.cm)**2 * np.pi  # Telescope area
texp = float(jsp['exposureTime'])*au.s  # Exposure time
mag_syst = 'Vega'  # Magnitude system
mag_band = 'V'  # Magnitude band
targ_mag = float(jsp['v'])  # Magnitude of the target @ 350 nm
bckg_mag = float(jsp['vSky'])  # Magnitude of the background @ 350 nm
airmass = float(jsp['airmass'])  # Airmass
pwv = 10.0  # Precipitable water vapor
moond = 0  # Days from new moon

spec_templ = 'qso'  # Function for the template spectrum ('flat', 'PL', 'qso', 'star')
spec_file = jsp['spectrum']
wmin = 295 * au.nm  # Minimum wavelength
wmax = 430 * au.nm  # Maximum wavelength
zem = None
igm_abs = None
extr_func = 'sum'  # Function for extracting the spectrum ('sum', 'opt' [very slow])
snr_sampl = 1*au.nm  # Data points per SNR point

phot_pars = ['bckg_mag', 'mag_syst', 'mag_band', 'targ_mag', 'texp']
spec_pars = ['spec_templ', 'spec_file', 'wmin', 'wmax', 'qso_zem', 'qso_lya_abs', 'airmass', 'pwv', 'moond']
psf_pars = ['psf_func', 'psf_sampl', 'psf_cen', 'targ_prof', 'targ_invrad_params', 'targ_sersic_params', 'slice_n', 
            'slice_length', 'slice_width', 'seeing']
ccd_pars = ['ccd_gain', 'ccd_ron', 'ccd_dark', 'ccd_xsize', 'ccd_ysize', 'pix_xsize', 'pix_ysize', 'ccd_xbin', 'ccd_ybin',
            'arm_n', 'wave_d', 'wave_d_shift', 'eff_file', 'eff_adc', 'eff_slc', 'eff_dch', 'eff_spc', 'eff_grt', 'eff_ccd', 
            'disp_file', 'disp_wave', 'disp_sampl', 'disp_resol', 'spat_scale', 'slice_gap', 'extr_func']