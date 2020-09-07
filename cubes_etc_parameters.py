#---------------------------------------
#user input
#---------------------------------------
"""SNRorExpT = 'ExpT'                                                      #Defines the type of input given. must be SNR or ExpT.
spectrum = 'database/Castelliap000T5250g45_N.flx'                       #star spectrum fiile
V = 12                                                                  #Object Vmag
wl = 3500                                                               #Wavelenght (Angstrom)
Vsky = 22                                                               #Sky V mag
slice_width = 0.25                                                      #arcsec
slice_height = 10                                                       #arcsec
seeing = 0.7                                                            #arcsec
airmass = 1.0                                                           #airmass
nbinx = 1                                                               #binning in x direction
nbiny = 1                                                               #Binning in y direction
expt = 2400                                                           #exposure time seconds"""
#---------------------------------------
#cubes parameters
#---------------------------------------
xk0 = 1                                                                  #Level of signal relative to the cont. (1=ct, <1 abs. line, >1 emiss. line)
dig = 79                                                                 #Echelle Groove density (g/mm)
alp = 63.435                                                             #Echelle angle of incidence measured from the facet (degrees)
del_= 63.435                                                             #Echelle blaze angle (deg)
fcam = 160                                                               #focal camera length (mm)
fno = 3.2                                                                #F/no nominal
T0 = -80                                                                 #Reference Temperature (C) for DC, i.e. T0
DC = 0.0005                                                              #Dark current e/sec/pixel at T0
T_det = -80                                                              #Working temperature (C)
FWC = 1000                                                               #Full Well Capacity (in e)
I_DR = 12                                                                #Dinamic Range (no bits)
ron = 2.5                                                                #read out noise  in e/pix
pixsize = 15                                                             #pixel size (micron)
dispscalefile ='database/DISPERSION_SCALE'                               #dispersion A/pixel 
ADC = True                                                               #ADC module inserted True/False
ADCeff = 'database/ADCTRAN'                                              #ADC (3mir, 4AR) Efficiency
SLICEReff='database/SLICERTRAN'                                          #Slicer (4MIR) Efficiency
DICHROICSeff='database/DICHROICSTRAN'                                    #Dichroics Efficiency
SPECTROGeff='database/SPECTROG_OPT'                                     #Spectrograph efficiency: OPT. Spec. (no grating no tel, detector)
GRATINGeff='database/GRATINGTRAN'                                        #Grating efficiency
CCDeff='database/DQER'                                                    #Detector QE (ASCII)
fileatm_tran='database/TRANS_ATM.dat'                                    #Atmosph. trans. (airmass=1) spectrum (ASCII)
mirr_refl='database/TEL'                                                 #Tot. reflec. for telescope (ASCII)
eps=0                                                                    #Telescope obstruction (0.143, cambiata con 0 per il file TEL) 
d_tel=8200                                                               #Telescope diameter (mm)
dw=0.0                                                                   #Wavelength bin in A
skyspectrum='database/SKY.dat'                                           #Sky  spectrum (ASCII)