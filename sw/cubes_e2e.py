# ------------------------------------------------------------------------------
# CUBES_E2E
# Simulate the spectral format of CUBES and estimate the SNR for an input spectrum
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------

from cubes_e2e_config import *
from astropy import units as au
from astropy import constants as ac
from astropy.io import ascii, fits
from astropy.modeling.fitting import LevMarLSQFitter as lm
from astropy.modeling.functional_models import Gaussian1D, Gaussian2D, Moffat2D, Sersic2D
from astropy.table import Table
import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import CubicSpline as cspline #interp1d
from scipy.interpolate import UnivariateSpline as uspline
from scipy.ndimage import gaussian_filter, interpolation
from scipy.signal import fftconvolve
from scipy.special import expit
import sys
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

class CCD(object):

    def __init__(self, psf, spec, xsize=ccd_xsize, ysize=ccd_ysize,
                 xbin=ccd_xbin, ybin=ccd_ybin,
                 pix_xsize=pix_xsize, pix_ysize=pix_ysize,
                 spat_scale=spat_scale, slice_n=slice_n, func=extr_func):
        self.psf = psf
        self.spec = spec
        self.xsize = xsize/xbin
        self.ysize = ysize/ybin
        self.xbin = xbin
        self.ybin = ybin
        self.npix = xbin*ybin
        self.pix_xsize = pix_xsize*xbin
        self.pix_ysize = pix_ysize*ybin
        self.spat_scale = spat_scale

        self.func = func
        
        if disp_file is not None:
            disp = ascii.read(disp_file)
            for a in range(arm_n):
                if arm_n==3: 
                    size = 'l' if ysize.value > 5000 else 's'
                else:
                    size = ''
                wave = np.array(disp['wave_'+str(arm_n)+'ch_'+str(a)+size])*0.1
                sampl = np.array(disp['sampl_'+str(arm_n)+'ch_'+str(a)+size])*1e3
                resol = np.array(disp['resol_'+str(arm_n)+'ch_'+str(a)+size])*1e3
                if a==0:
                    self.disp_wave = [wave]
                    self.disp_sampl = [sampl]
                    self.disp_resol = [resol]
                else:
                    self.disp_wave = np.vstack((self.disp_wave, wave))
                    self.disp_sampl = np.vstack((self.disp_sampl, sampl))
                    self.disp_resol = np.vstack((self.disp_resol, resol))

            self.disp_wave *= au.nm
            self.disp_sampl *= au.nm/au.pixel
            #print(self.disp_wave)


    def add_arms(self, n, wave_d, wave_d_shift):
        self.xcens = [self.xsize.value//2]*n
        
        self.signal = np.zeros((int(self.ysize.value), int(self.xsize.value), n))
        self.noise = np.zeros((int(self.ysize.value), int(self.xsize.value), n))
        self.targ_noise = np.zeros((int(self.ysize.value), int(self.xsize.value), n))

        self.dark = np.sqrt((ccd_dark*ccd_gain*self.npix*self.spec.phot.texp)\
                            .to(au.photon).value)
        self.ron = (ccd_ron*ccd_gain).value

        if arm_n > 1:
            wmax = wave_d[0]+wave_d_shift
            wmin = wmax-self.ysize*np.mean(self.disp_sampl[0])

            dw = np.full(int(self.ysize.value), 0.5*(wmin.value+wmax.value))
            w = np.ravel(self.disp_wave.value)
            s = np.ravel(self.disp_sampl)*ccd_ybin
            w = w[np.argsort(w)]
            s = s[np.argsort(w)]
            for j in range(10):
                wmin2 = wmax.value-np.sum(cspline(w, s)(dw))
                dw = np.linspace(wmin2, wmax.value, int(self.ysize.value))
            wmin = wmin2*wmin.unit 
                
            self.wmins = np.array([wmin.to(au.nm).value])
            self.wmaxs = np.array([wmax.to(au.nm).value])
            self.wmins_d = np.array([wmin.to(au.nm).value])
            self.wmaxs_d = np.array([wave_d[0].to(au.nm).value])
            for i in range(len(wave_d)): 
                wmin = wave_d[i]-wave_d_shift
                wmax = wmin+self.ysize*np.mean(self.disp_sampl[i+1])

                dw = np.full(int(self.ysize.value), 0.5*(wmin.value+wmax.value))
                for j in range(10):
                    wmax2 = wmin.value+np.sum(cspline(w, s)(dw))
                    dw = np.linspace(wmin.value, wmax2, int(self.ysize.value))
                wmax = wmax2*wmax.unit
               
                self.wmins = np.append(self.wmins, wmin.to(au.nm).value)
                self.wmaxs = np.append(self.wmaxs, wmax.to(au.nm).value)
                self.wmins_d = np.append(self.wmins_d, wave_d[i].to(au.nm).value)
                try:
                    self.wmaxs_d = np.append(self.wmaxs_d, wave_d[i+1].to(au.nm).value)
                except:
                    self.wmaxs_d = np.append(self.wmaxs_d, wmax.to(au.nm).value)
        else:
            wcen = 347.5*au.nm
            wmin = wcen-self.ysize.value//2*self.ysize.unit*self.disp_sampl[0]
            wmax = wcen+self.ysize.value//2*self.ysize.unit*self.disp_sampl[-1]
            dwmin = np.full(int(self.ysize.value), wcen)
            dwmax = np.full(int(self.ysize.value), wcen)
            for j in range(10):
                wmin2 = wcen.value-np.sum(cspline(np.ravel(self.disp_wave.value), np.ravel(self.disp_sampl)*ccd_ybin)(dwmin))/2
                wmax2 = wcen.value+np.sum(cspline(np.ravel(self.disp_wave.value), np.ravel(self.disp_sampl)*ccd_ybin)(dwmax))/2
                dwmin = np.linspace(wmin2, wcen.value, int(self.ysize.value))
                dwmax = np.linspace(wcen.value, wmax2, int(self.ysize.value))

            wmin = wmin2*wmin.unit 
            wmax = wmax2*wmax.unit 
                
            self.wmins = np.array([wmin.to(au.nm).value])
            self.wmaxs = np.array([wmax.to(au.nm).value])
            self.wmins_d = np.array([wmin.to(au.nm).value])
            self.wmaxs_d = np.array([wmax.to(au.nm).value])

        self.wmins_d[0] = 200
        self.wmaxs_d[-1] = 500
        self.wmins = self.wmins * au.nm
        self.wmaxs = self.wmaxs * au.nm
        self.wmins_d = self.wmins_d * au.nm
        self.wmaxs_d = self.wmaxs_d * au.nm

        #print(self.wmins)
        #print(self.wmaxs)

        
        self.mod_init = []
        self.sl_cen = []
        self.spec.m_d = []
        self.spec.M_d = []
        self.spec.arm_wave = []
        self.spec.arm_targ = []
        self.spec.tot_eff = self.tot_eff
        self.targ_sum = 0
        self.bckg_sum = 0
        self.targ_prof = []
        self.bckg_prof = []
        
        self.targ_noise_max = []
        self.bckg_noise_med = []
        self.spec.fwhm = []
        self.spec.resol = []

        self.eff_wave = []
        self.eff_fib = []
        self.eff_opo = []
        self.eff_adc = []
        self.eff_slc = []
        self.eff_dch = []
        self.eff_fld = []
        self.eff_col = []
        self.eff_cam = []
        self.eff_grt = []
        self.eff_ccd = []
        self.eff_tel = []
        self.eff_tot = []

        for i, (x, m, M, m_d, M_d) in enumerate(zip(
            self.xcens, self.wmins, self.wmaxs, self.wmins_d, self.wmaxs_d)):
            self.sl_targ_prof = []
            self.sl_bckg_prof = []
            self.arm_counter = i
            self.arm_range = np.logical_and(self.spec.wave.value>m.value,
                                            self.spec.wave.value<M.value)
            self.arm_wave = self.spec.wave[self.arm_range].value
            self.arm_targ = self.spec.targ_conv[self.arm_range].value
            xlength = int(slice_length/self.spat_scale/self.pix_xsize)
            self.sl_hlength = xlength // 2
            self.psf_xlength = int(np.ceil(self.psf.seeing/self.spat_scale
                                           /self.pix_xsize))
            xspan = xlength + int(slice_gap.value/self.xbin)
            xshift = (slice_n*xspan+xlength)//2
            self.add_slices(int(x), xshift, xspan, self.psf_xlength,
                            wmin=m.value, wmax=M.value, wmin_d=m_d.value,
                            wmax_d=M_d.value)
            self.spec.m_d.append(m_d.value)
            self.spec.M_d.append(M_d.value)


            self.spec.arm_wave.append(self.arm_wave)
            self.spec.arm_targ.append(self.arm_targ)
             
            spl_wave = self.disp_wave[i]
            spl_sampl = self.disp_sampl[i]
            spl_resol = np.array(self.disp_resol[i])

            disp = cspline(spl_wave, spl_resol*spl_sampl*ccd_ybin)(self.arm_wave)
            self.spec.fwhm.append(self.arm_wave/disp)
            self.spec.resol.append(cspline(spl_wave, spl_resol)(self.arm_wave))
            
            self.targ_prof = np.append(self.targ_prof, self.sl_targ_prof)
            self.bckg_prof = np.append(self.bckg_prof, self.sl_bckg_prof)
            self.sl_targ_peak = self.sl_targ_prof[self.sl_targ_prof>99e-2*np.max(self.sl_targ_prof)]
            self.sl_targ_prof = self.sl_targ_prof * au.ph
            self.sl_bckg_prof = self.sl_bckg_prof * au.ph
            
            self.targ_noise_max = np.append(self.targ_noise_max, np.sqrt(np.max(self.sl_targ_prof.value)))
            self.bckg_noise_med = np.append(self.bckg_noise_med, np.sqrt(np.median(self.sl_bckg_prof.value)))

            
        print("Slices projected onto arms.       ")

        self.targ_peak = self.targ_prof[self.targ_prof>99e-2*np.max(self.targ_prof)]

        self.targ_prof = self.targ_prof * au.ph
        self.bckg_prof = self.bckg_prof * au.ph

        self.targ_sum = self.targ_sum * au.ph
        self.bckg_sum = self.bckg_sum * au.ph

        self.spec.targ_sum = self.targ_sum
        
        self.targ_noise_max = self.targ_noise_max * au.ph/au.pixel
        self.bckg_noise_med = self.bckg_noise_med * au.ph/au.pixel

        
    def add_slice(self, trace, trace_targ, trace_bckg, xcen, wmin, wmax, wmin_d, wmax_d):

        wave_in = self.spec.wave.value
        
        # Normalization 
        targ_sum = np.sum(self.spec.targ_conv)
        bckg_sum = np.sum(self.spec.bckg_conv)
        targ_red = self.spec.targ_conv[np.logical_and(wave_in>self.wmins[0].value, wave_in<self.wmaxs[-1].value)]
        bckg_red = self.spec.bckg_conv[np.logical_and(wave_in>self.wmins[0].value, wave_in<self.wmaxs[-1].value)]
        wave_red = wave_in[np.logical_and(wave_in>self.wmins[0].value, wave_in<self.wmaxs[-1].value)]
        self.spec.targ_norm = targ_red/targ_sum #np.sum(targ)
        self.spec.bckg_norm = bckg_red/bckg_sum #np.sum(bckg)

        wave = self.wave_grid(wmin, wmax)
        
        # Apply correct sampling
        sampl = cspline(self.disp_wave[self.arm_counter], self.disp_sampl[self.arm_counter]*ccd_ybin)(wave)
        wave = wmin+np.cumsum(sampl)
        #targ = cspline(wave_red, targ_red)(wave)*targ_red.unit
        #bckg = cspline(wave_red, bckg_red)(wave)*bckg_red.unit
        targ = np.interp(wave, wave_red, targ_red)*targ_red.unit
        bckg = np.interp(wave, wave_red, bckg_red)*bckg_red.unit

        
        # Since self.spec.targ/bckg_norm were normalized, we have to normalize targ/bckg again after csplining
        targ_arm = targ_red[np.logical_and(wave_red>wmin, wave_red<wmax)]
        bckg_arm = bckg_red[np.logical_and(wave_red>wmin, wave_red<wmax)]
        targ = targ/np.sum(targ)*np.sum(targ_arm)/targ_sum
        bckg = bckg/np.sum(bckg)*np.sum(bckg_arm)/bckg_sum
            
            
        sl_trace = self.rebin(trace, self.sl_hlength*2)
        sl_trace_targ = self.rebin(trace_targ, self.sl_hlength*2)
        sl_trace_bckg = self.rebin(trace_bckg, self.sl_hlength*2)
        sl_targ = self.rebin(targ.value, self.ysize.value)
        sl_bckg = self.rebin(bckg.value, self.ysize.value)
        
        # Efficiency
        if wmin_d is not None and wmax_d is not None:
            tot_eff = self.tot_eff(wave, wmin_d, wmax_d)
            sl_targ = sl_targ * tot_eff
            sl_bckg = sl_bckg * tot_eff

        sl_targ_prof = np.multiply.outer(sl_targ, sl_trace_targ)
        sl_bckg_prof = np.multiply.outer(sl_bckg, sl_trace_bckg)
        self.targ_sum += np.sum(sl_targ_prof)
        self.bckg_sum += np.sum(sl_bckg_prof)
        self.sl_targ_prof = np.append(self.sl_targ_prof, sl_targ_prof)
        self.sl_bckg_prof = np.append(self.sl_bckg_prof, sl_bckg_prof)

        signal = np.round(sl_targ_prof+sl_bckg_prof)

        targ_noise = np.random.normal(0., 1., sl_targ_prof.shape)*np.sqrt(np.abs(sl_targ_prof))
        bckg_noise = np.random.normal(0., 1., sl_bckg_prof.shape)*np.sqrt(np.abs(sl_bckg_prof))
        dsignal = targ_noise+bckg_noise
        noise = np.sqrt(targ_noise**2 + bckg_noise**2 + self.dark**2 + self.ron**2)
        self.signal[:,xcen-self.sl_hlength:xcen+self.sl_hlength][:,:,self.arm_counter] = np.round(signal+dsignal)
        self.noise[:,xcen-self.sl_hlength:xcen+self.sl_hlength][:,:,self.arm_counter] = noise
        self.targ_noise[:,xcen-self.sl_hlength:xcen+self.sl_hlength][:,:,self.arm_counter] = targ_noise

        return sl_trace, sl_targ, np.mean(signal)


    def add_slices(self, xcen, xshift, xspan, psf_xlength, wmin, wmax, wmin_d,
                   wmax_d):

        xcens = range(xcen-xshift, xcen+xshift, xspan)

        for s, (c, t, t_t, t_b) in enumerate(zip(xcens[1:], self.psf.traces, self.psf.traces_targ, self.psf.traces_bckg)):
            print("Projecting slice %i onto arm %i..." % (s, self.arm_counter), end='\r')

            _, _, sl_msignal = self.add_slice(t, t_t, t_b, c, wmin, wmax, wmin_d, wmax_d)

            self.mod_init.append(
                Gaussian1D(amplitude=sl_msignal, mean=c, stddev=psf_xlength))
            self.sl_cen.append(c)
            

    def draw(self):
        fig_p, self.ax_p = plt.subplots(figsize=(10,7))
        self.ax_p.set_title("Photon balance (CCD)")
        sl_v = np.array([self.spec.targ_tot.value, np.sum(self.psf.targ_slice.value), 
                         np.sum(self.psf.z_targ.value)-np.sum(self.psf.targ_slice.value),
                         self.targ_sum.value, np.sum(self.psf.z_targ.value)-self.targ_sum.value])
        sl_l = ["target: %2.3e %s"  % (sl_v[0], self.spec.targ_tot.unit), 
                "on slit: %2.3e %s"  % (sl_v[1], self.psf.z_targ.unit), 
                "off slit: %2.3e %s"  % (sl_v[2], self.psf.z_targ.unit),
                "on CCD: %2.3e %s"  % (sl_v[3], self.targ_sum.unit), 
                "off CCD: %2.3e %s"  % (sl_v[4], self.targ_sum.unit)] 
        sl_c = ['C0', 'C0', 'C0', 'C0', 'C0']
        p0 = self.ax_p.pie([sl_v[0]], colors=[sl_c[0]], startangle=90, radius=1,
                      wedgeprops=dict(width=0.2, edgecolor='w'))
        p1 = self.ax_p.pie(sl_v[1:3], colors=sl_c[1:3], startangle=90, radius=0.8,
                           wedgeprops=dict(width=0.2, edgecolor='w'))
        p2 = self.ax_p.pie(sl_v[3:], colors=sl_c[3:], autopct='%1.1f%%', startangle=90, radius=0.6, 
                           wedgeprops=dict(width=0.2, edgecolor='w'))
        p1[0][0].set_alpha(2/3)
        p1[0][1].set_alpha(1/6)
        p2[0][0].set_alpha(1/3)
        p2[0][1].set_alpha(1/6)
        self.ax_p.legend([p0[0][0],p1[0][0],p2[0][0]], [sl_l[0]]+sl_l[1::2])

        fig_r, self.ax_r = plt.subplots(figsize=(5,5))
        self.ax_r.set_title("Pixel size")
        xsize = pix_xsize*ccd_xbin
        ysize = pix_ysize*ccd_ybin
        xreal = pix_xsize*ccd_xbin*spat_scale
        yreal = np.mean(self.disp_sampl.value)*ccd_ybin*self.disp_sampl.unit*au.pixel
        size = max(xsize.value, ysize.value)
        
        self.ax_r.add_patch(patches.Rectangle((0,0), xsize.value, ysize.value, edgecolor='b', facecolor='b', alpha=0.3))
        self.ax_r.set_xlim(-0.2*size, size*1.2)
        self.ax_r.set_ylim(-0.2*size, size*1.2)
        for x in np.arange(0.0, xsize.value, pix_xsize.value):
            self.ax_r.axvline(x, 1/7, 1/7+5/7*ysize.value/size, linestyle=':')
        for y in np.arange(0.0, ysize.value, pix_ysize.value):
            self.ax_r.axhline(y, 1/7, 1/7+5/7*xsize.value/size, linestyle=':')
        self.ax_r.text(0.5*xsize.value, -0.1*ysize.value, 
                       '%3.1f %s = %3.1f %s' % (xsize.value, xsize.unit, xreal.value, xreal.unit), ha='center', va='center')
        self.ax_r.text(-0.1*xsize.value, 0.5*ysize.value,
                       '%3.1f %s ~ %3.2e %s' % (ysize.value, ysize.unit, yreal.value, yreal.unit), ha='center', va='center',
                       rotation=90)

        self.ax_r.set_axis_off()

        
        fig_s, self.ax_s = plt.subplots(3, 1, figsize=(10,10), sharex=True)
        self.ax_s[0].set_title("Resolution and sampling")
        for i in range(arm_n):
            self.ax_s[0].plot(self.spec.arm_wave[i], self.spec.resol[i], label='Arm %i' % i, color='C0', alpha=1-i/arm_n)
            self.ax_s[0].get_xaxis().set_visible(False)
            self.ax_s[0].set_ylabel('Resolution')

            spl_wave = self.disp_wave[i]
            spl_sampl = self.disp_sampl[i]
            self.ax_s[1].plot(self.spec.arm_wave[i], cspline(spl_wave, spl_sampl*ccd_ybin)(self.spec.arm_wave[i]), 
                              label='Arm %i' % i, color='C0', alpha=1-i/arm_n)

            self.ax_s[1].get_xaxis().set_visible(False)
            self.ax_s[1].set_ylabel('Sampling (%s/%s)' % (au.nm, au.pixel))
            self.ax_s[2].plot(self.spec.arm_wave[i], self.spec.fwhm[i], label='Arm %i' % i, color='C0', alpha=1-i/arm_n)
            self.ax_s[2].set_xlabel('Wavelength (%s)' % au.nm)
            self.ax_s[2].set_ylabel('FWHM (%s)' % au.pixel)
        self.ax_s[2].axhline(2.0, linestyle=':')
        self.ax_s[2].text(np.min(self.spec.arm_wave[0]), 2.0, "Nyquist limit", ha='left', va='bottom')
        self.ax_s[2].legend()
        fig_s.subplots_adjust(hspace=0)
        
        fig_e, self.ax_e = plt.subplots(figsize=(10,7))
        self.ax_e.set_title("Efficiency")
        for i in range(0,arm_n*slice_n,slice_n):  # Only one slice per arm is plotted, as they are the same efficiency-wise
            self.ax_e.plot(self.eff_wave[i], self.eff_fib[i], label='Fiber' if i==0 else '', color='C0', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_adc[i], label='ADC' if i==0 else '', color='C1', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_opo[i], label='Other pre-optics' if i==0 else '', color='C2', 
                           linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_slc[i], label='Slicer' if i==0 else '', color='C3', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_dch[i], label='Dichroich' if i==0 else '', color='C4', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_fld[i], label='Folding' if i==0 else '', color='C5', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_col[i], label='Collimator' if i==0 else '', color='C6', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_cam[i], label='Camera' if i==0 else '', color='C7', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_grt[i], label='Grating' if i==0 else '', color='C8', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_ccd[i], label='CCD' if i==0 else '', color='C9', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_tel[i], label='Telescope' if i==0 else '', color='black', linestyle=':')
            self.ax_e.plot(self.eff_wave[i], self.eff_tot[i], label='Total' if i==0 else '', color='C0')
#            self.ax_e.plot(self.eff_wave[i], self.eff_tot[i]*cspline(self.spec.wave, self.spec.atmo_ex)(self.eff_wave[i]),
            self.ax_e.plot(self.eff_wave[i], self.eff_tot[i]*np.interp(self.eff_wave[i], self.spec.wave, self.spec.atmo_ex),
                           label='Total including extinction' if i==0 else '', color='C0', linestyle='--')
            
            self.ax_e.set_xlabel('Wavelength (%s)' % au.nm)
            self.ax_e.set_ylabel('Fractional throughput')
        self.ax_e.legend()

        
        

        fig_b, self.ax_b = plt.subplots(figsize=(7,7))
        self.ax_b.set_title("Noise breakdown")
        x = np.arange(arm_n)
        self.ax_b.bar(x-0.3, self.targ_noise_max.value, width=0.2, label='Target (maximum)')
        self.ax_b.bar(x-0.1, self.bckg_noise_med.value, width=0.2, label='Background (median)')
        self.ax_b.bar(x+0.1, self.dark, width=0.2, label='Dark')
        self.ax_b.bar(x+0.3, self.ron, width=0.2, label='Read-out')
        self.ax_b.set_xlabel('Arm')
        self.ax_b.set_ylabel('Noise (%s)' % self.targ_noise_max.unit)
        self.ax_b.set_xticks(range(arm_n))
        self.ax_b.legend()
        for i in range(arm_n):
            fig, self.ax = plt.subplots(figsize=(8,8))
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
                        
            im = self.ax.imshow(self.signal[:,:,i], vmin=0)
                        
            self.ax.set_title('CCD')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            cax.xaxis.set_label_position('top')
            cax.set_xlabel(au.ph)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
            
            
    def extr_arms(self, n, slice_n):
        wave_snr = np.arange(self.wmins[0].value, self.wmaxs[-1].value, snr_sampl.value)
        self.spec.wave_snr = wave_snr
        
        for a in range(n):
            wave_extr = self.wave_grid(self.wmins[a], self.wmaxs[a])
            
            # Apply correct sampling
            sampl = cspline(self.disp_wave[a], self.disp_sampl[a]*ccd_ybin)(wave_extr)
            wave_extr = self.wmins[a]+np.cumsum(sampl)*self.wmins[a].unit


            flux_extr = 0
            err_extr = 0
            err_targ_extr = 0
            err_bckg_extr = 0
            err_dark_extr = 0
            err_ron_extr = 0
            pix_extr = 0

            b = []
            for s in range(slice_n):
                print("Extracting slice %i from arm %i..." % (s, a), end='\r')
                i = a*slice_n+s
                x = range(self.sl_cen[i]-self.sl_hlength,
                          self.sl_cen[i]+self.sl_hlength)
                s_extr = np.empty(int(self.ysize.value))
                n_extr = np.empty(int(self.ysize.value))
                n_targ_extr = np.empty(int(self.ysize.value))
                n_bckg_extr = np.empty(int(self.ysize.value))
                n_dark_extr = np.empty(int(self.ysize.value))
                n_ron_extr = np.empty(int(self.ysize.value))                
                p_extr = np.empty(int(self.ysize.value))                
                for p in range(int(self.ysize.value)):
                    row = self.signal[p, :, a]
                    y = row[self.sl_cen[i]-self.sl_hlength:self.sl_cen[i]+self.sl_hlength]
                    b1 = np.median(row[self.sl_cen[i]-self.sl_hlength+1:self.sl_cen[i]-self.sl_hlength+6])
                    b2 = np.median(row[self.sl_cen[i]+self.sl_hlength-6:self.sl_cen[i]+self.sl_hlength-1])

                    b.append(0.5*(b1+b2))
                    y = y - b[-1]
                    dy = self.noise[p, self.sl_cen[i]-self.sl_hlength:
                                      self.sl_cen[i]+self.sl_hlength, a]
                    dy_targ = self.targ_noise[p, self.sl_cen[i]-self.sl_hlength:
                                      self.sl_cen[i]+self.sl_hlength, a]
                    s_extr[p], n_extr[p], (n_targ_extr[p], n_bckg_extr[p], n_dark_extr[p], n_ron_extr[p], p_extr[p]) \
                        = getattr(self, 'extr_'+self.func)(y, dy=dy, dy_targ=dy_targ, mod=self.mod_init[i], x=x, p=p)
                flux_extr += s_extr
                err_extr = np.sqrt(err_extr**2 + n_extr**2)
                err_targ_extr = np.sqrt(err_targ_extr**2 + n_targ_extr**2)
                err_bckg_extr = np.sqrt(err_bckg_extr**2 + n_bckg_extr**2)
                err_dark_extr = np.sqrt(err_dark_extr**2 + n_dark_extr**2)
                err_ron_extr = np.sqrt(err_ron_extr**2 + n_ron_extr**2)
                
                pix_extr = pix_extr+p_extr
            print("SNR across arm %i:               " % a)
            snr_extr = flux_extr/err_extr
            for w in self.disp_wave[a]:
                w_near = wave_extr.flat[np.abs(wave_extr.value - w.value).argmin()]
                #print(" %3.2f %s: %3.0f" % (w.value, w.unit, cspline(wave_extr, snr_extr)(w)))
                print(" %3.2f %s: %3.0f" % (w.value, w.unit, snr_extr[wave_extr==w_near]))


            dw = (wave_extr[2:]-wave_extr[:-2])*0.5
            dw = np.append(dw[:1], dw)
            dw = np.append(dw, dw[-1:])
            flux_extr = flux_extr / dw
            err_extr = err_extr / dw

            from scipy.signal import savgol_filter
            err_savgol = savgol_filter(err_extr, len(wave_extr)//len(wave_snr)//2*4+1, 3)

            snr_extr = flux_extr/err_savgol
            snr_extr[np.where(np.isnan(snr_extr))] = 0
            snr_extr[np.where(np.isinf(snr_extr))] = 0
            snr_spl = cspline(wave_extr, snr_extr)(wave_snr)
            snr_spl[wave_snr<self.wmins[a].value] = 0.0
            snr_spl[wave_snr>self.wmaxs[a].value] = 0.0
            if a == 0:
                self.spec.snr = snr_spl
            else:
                self.spec.snr = np.sqrt(self.spec.snr**2+snr_spl**2)
            self.spec.wave_snr_2 = wave_extr
            self.spec.snr_2 = snr_extr

            if a==0:
                self.spec.wave_extr = np.array(wave_extr)
                self.spec.flux_extr = np.array(flux_extr)
                self.spec.err_extr = np.array(err_extr)
                self.spec.err_targ_extr = np.array(err_targ_extr)
                self.spec.err_bckg_extr = np.array(err_bckg_extr)
                self.spec.err_dark_extr = np.array(err_dark_extr)
                self.spec.err_ron_extr = np.array(err_ron_extr)
                self.spec.pix_extr = np.array(pix_extr)

            else:
                self.spec.wave_extr = np.vstack((self.spec.wave_extr, wave_extr.value))
                self.spec.flux_extr = np.vstack((self.spec.flux_extr, flux_extr.value))
                self.spec.err_extr = np.vstack((self.spec.err_extr, err_extr.value))
                self.spec.err_targ_extr = np.vstack((self.spec.err_targ_extr, err_targ_extr))
                self.spec.err_bckg_extr = np.vstack((self.spec.err_bckg_extr, err_bckg_extr))
                self.spec.err_dark_extr = np.vstack((self.spec.err_dark_extr, err_dark_extr))
                self.spec.err_ron_extr = np.vstack((self.spec.err_ron_extr, err_ron_extr))
                self.spec.pix_extr = np.vstack((self.spec.pix_extr, pix_extr))
            
        self.spec.flux_extr = self.spec.flux_extr * au.ph
        if arm_n > 1:
            self.spec.flux_final_tot = np.sum([np.sum(f)/len(f) * (M-m) for f, M, m 
                                               in zip(self.spec.flux_extr.value, self.wmaxs.value, self.wmins.value)])
        else:
            self.spec.flux_final_tot = np.sum(self.spec.flux_extr.value)/len(self.spec.flux_extr.value) \
                                              * (self.wmaxs.value-self.wmins.value)
            
        self.spec.flux_final_tot = self.spec.flux_final_tot * au.ph

        
    def extr_sum(self, y, dy, dy_targ, **kwargs):
        sel = np.s_[self.sl_hlength-self.psf_xlength
                    :self.sl_hlength+self.psf_xlength]
        ysel = y[sel]
        dysel = dy[sel]
        dysel_targ = dy_targ[sel]
        dysel_bckg = np.sqrt(dysel**2 - dysel_targ**2 - self.dark**2 - self.ron**2)

        s = np.sum(ysel)
        n = np.sqrt(np.sum(dysel**2))
        n_targ = np.sqrt(np.sum(dysel_targ**2))
        n_bckg = np.sqrt(np.sum(dysel_bckg**2))
        pix = len(ysel)
        n_dark = np.sqrt(pix)*self.dark
        n_ron = np.sqrt(pix)*self.ron

        if np.isnan(s) or np.isnan(n) or np.isinf(s) or np.isinf(n) \
            or np.abs(s) > 1e30 or np.abs(n) > 1e30:
            s = 0
            n = 1
            n_targ = 1
            n_bckg = 0
            n_dark = 0
            n_ron = 0
        return s, n, (n_targ, n_bckg, n_dark, n_ron, pix)

    
    def extr_opt(self, y, dy, dy_targ, mod, x, p):
        mod_fit = lm()(mod, x, y)(x)
        mod_fit[mod_fit < 1e-3] = 0
        if np.sum(mod_fit*dy) > 0 and not np.isnan(mod_fit).any():
            mod_norm = mod_fit/np.sum(mod_fit)
            w = dy>0
            s = np.sum(mod_norm[w]*y[w]/dy[w]**2)/np.sum(mod_norm[w]**2/dy[w]**2)
            n = np.sqrt(np.sum(mod_norm[w])/np.sum(mod_norm[w]**2/dy[w]**2))
            n_targ = np.sqrt(np.sum(mod_norm[w])/np.sum(mod_norm[w]**2/dy_targ[w]**2))
            n_bckg = np.sqrt(n**2 - n_targ**2 - self.dark**2 - self.ron**2)
            pix = np.sum(mod_norm[w])
            n_dark = np.sqrt(pix)*self.dark
            n_ron = np.sqrt(pix)*self.ron
        else:
            s = 0
            n = 1
            n_targ = 1
            n_bckg = 0
            n_dark = 0
            n_ron = 0
            pix = 0
        if np.isnan(s) or np.isnan(n) or np.isinf(s) or np.isinf(n) \
            or np.abs(s) > 1e30 or np.abs(n) > 1e30:
            s = 0
            n = 1
            n_targ = 1
            n_bckg = 0
            n_dark = 0
            n_ron = 0
        return s, n, (n_targ, n_bckg, n_dark, n_ron, pix)

    
    def rebin(self, arr, length):
        # Adapted from http://www.bdnyc.org/2012/03/rebin-numpy-arrays-in-python/
        # Good for now, but need to find more sophisticated solution
        zoom_factor = length / arr.shape[0]
        new = interpolation.zoom(arr, zoom_factor)
        if np.sum(new) != 0:
            return new/np.sum(new)*np.sum(arr)
        else:
            return new


    def tot_eff(self, wave, wmin_d, wmax_d, fact=2, method='interp'):
        dch_shape = expit(fact*(wave-wmin_d))*expit(fact*(wmax_d-wave))
        i = self.arm_counter
        if eff_file is not None:
            eff = ascii.read(eff_file)
            eff_wave = np.array(eff['wavelength'])*0.1*au.nm
            eff_fib = np.array(eff['fiber'])
            eff_adc = np.array(eff['adc'])
            eff_opo = np.array(eff['other_preopt'])
            eff_slc = np.array(eff['slicer'])
            eff_dch = np.array(eff['dichroic'])
            eff_fld = np.array(eff['fold'])
            eff_col = np.array(eff['collimator'])
            eff_cam = np.array(eff['camera'])
            
            eff_grt = np.array(eff['grating_'+str(arm_n)+'ch'])
            eff_ccd = np.array(eff['qe'])
            eff_tel = np.array(eff['telescope'])
            
        if method=='cspline':
            fib = cspline(eff_wave, eff_fib)(wave)
            adc = cspline(eff_wave, eff_adc)(wave)
            opo = cspline(eff_wave, eff_opo)(wave)
            slc = cspline(eff_wave, eff_slc)(wave)
            dch = cspline(eff_wave, eff_dch)(wave) * dch_shape
            fld = cspline(eff_wave, eff_fld)(wave)
            col = cspline(eff_wave, eff_col)(wave)
            cam = cspline(eff_wave, eff_cam)(wave)
            grt = cspline(eff_wave, eff_grt)(wave)
            ccd = cspline(eff_wave, eff_ccd)(wave)
            tel = cspline(eff_wave, eff_tel)(wave)
        elif method=='interp':
            fib = np.interp(wave, eff_wave, eff_fib)
            adc = np.interp(wave, eff_wave, eff_adc)
            opo = np.interp(wave, eff_wave, eff_opo)
            slc = np.interp(wave, eff_wave, eff_slc)
            dch = np.interp(wave, eff_wave, eff_dch) * dch_shape
            fld = np.interp(wave, eff_wave, eff_fld)
            col = np.interp(wave, eff_wave, eff_col)
            cam = np.interp(wave, eff_wave, eff_cam)
            grt = np.interp(wave, eff_wave, eff_grt)
            ccd = np.interp(wave, eff_wave, eff_ccd)
            tel = np.interp(wave, eff_wave, eff_tel)
        tot = fib * adc * opo * slc * dch * fld * col * cam * grt * ccd * tel
        
        self.eff_wave.append(wave)
        self.eff_fib.append(fib)
        self.eff_adc.append(adc)
        self.eff_opo.append(opo)
        self.eff_slc.append(slc)
        self.eff_dch.append(dch)
        self.eff_fld.append(fld)
        self.eff_col.append(col)
        self.eff_cam.append(cam)
        self.eff_grt.append(grt)
        self.eff_ccd.append(ccd)
        self.eff_tel.append(tel)
        self.eff_tot.append(tot)

        return tot

    
    def wave_grid(self, wmin, wmax):
        return np.linspace(wmin, wmax, int(self.ysize.value))


class Photons(object):

    def __init__(self, targ_mag=targ_mag, bckg_mag=bckg_mag, area=area,
                 texp=texp):
        self.targ_mag = targ_mag
        self.bckg_mag = bckg_mag
        self.area = area
        self.texp = texp
        self.wave_ref = globals()['wave_ref_'+mag_syst][mag_band]
        self.flux_ref = globals()['flux_ref_'+mag_syst][mag_band]
        
        try:
            data_band = ascii.read('../database/phot_%s.dat' % mag_band)
            self.wave_band = data_band['col1'] * au.nm
            if mag_band in ['J', 'H', 'K']:
                self.wave_band = self.wave_band*1e3
            self.dwave_band = self.wave_band[1]-self.wave_band[0]
            self.flux_band = data_band['col2']
            self.flux_band = self.flux_band/np.sum(self.flux_band)*self.dwave_band
        except:
            print("Band not found.")
            
        self.data_u = ascii.read('../database/phot_U.dat')
        self.wave_u = self.data_u['col1'] * au.nm
        self.dwave_u = self.wave_u[1]-self.wave_u[0]
        self.flux_u = self.data_u['col2']
        self.flux_u = self.flux_u/np.sum(self.flux_u)*self.dwave_u

        f = self.flux_ref * self.area * texp  # Flux density @ 555.6 nm, V = 0
        self.targ = f * pow(10, -0.4*self.targ_mag)
        self.bckg = f * pow(10, -0.4*self.bckg_mag) / au.arcsec**2

        self.atmo()
        print("Photons collected.")

        
    def atmo(self):
        data = fits.open('../database/atmoexan.fits')[1].data
        self.atmo_wave = data['LAMBDA']*0.1 * au.nm
        self.atmo_ex = data['LA_SILLA']


class PSF(object):

    def __init__(self, spec, seeing=seeing, slice_width=slice_width,
                 xsize=slice_length, ysize=slice_length, sampl=psf_sampl,
                 func=psf_func):
        self.spec = spec
        self.seeing = seeing
        self.slice_width = slice_width
        self.xsize = xsize
        self.ysize = ysize
        self.area = xsize * ysize
        self.sampl = sampl
        self.func = func
        self.rects = []
        x = np.linspace(-xsize.value/2, xsize.value/2, int(sampl.value))
        y = np.linspace(-ysize.value/2, ysize.value/2, int(sampl.value))
        self.x, self.y = np.meshgrid(x, y)

        getattr(self, func)()  # Apply the chosen function for the PSF
        prof_sel = np.ones(self.z.shape, dtype=bool)
        if targ_prof is not None:
            z, prof_sel = getattr(self, targ_prof)()
            self.z = fftconvolve(self.z, z, mode='same')#[psf_cen[1]-9:psf_cen[1]+9,psf_cen[0]-9:psf_cen[0]+9])
            #print(self.z)
            
        self.z_norm_sel_ratio = np.sum(prof_sel)/np.size(self.z)
        #print(self.z_norm_sel_ratio)
        self.z_norm = self.z/np.sum(self.z[prof_sel])

        self.z_norm_a = np.ones(self.z.shape)/self.z.size
        self.z_targ = self.z_norm * self.spec.targ_tot  # Counts from the target
        self.z_bckg = self.z_norm_a * self.spec.bckg_tot*self.area  # Counts from the background
        self.z = self.z_targ + self.z_bckg
        #print(self.z)

        self.spec.z_targ = self.z_targ
        
        
    def add_slice(self, cen=None):  # Central pixel of the slice

        width = self.slice_width
        length = self.ysize
        if cen == None:
            cen = (0, 0)
        hwidth = width.value / 2
        hlength = length.value / 2

        rect = patches.Rectangle((cen[0]-hwidth, cen[1]-hlength), width.value,
                                 length.value, edgecolor='r', facecolor='none')
        self.rects.append(rect)

        # In this way, fractions of pixels in the slices are counted
        ones = np.ones(self.x.shape)
        self.pix_xsize = self.xsize / self.sampl
        self.pix_ysize = self.ysize / self.sampl
        left_dist = (self.x-cen[0]+hwidth) / self.pix_xsize.value
        right_dist = (cen[0]+hwidth-self.x) / self.pix_xsize.value
        down_dist = (self.y-cen[1]+hlength) / self.pix_ysize.value
        up_dist = (cen[1]+hlength-self.y) / self.pix_ysize.value
        
        # This mask gives half weight to the edge pixels, to avoid superposition issues
        mask_left = np.maximum(-ones, np.minimum(ones, left_dist))*0.5+0.5
        mask_right = np.maximum(-ones, np.minimum(ones, right_dist))*0.5+0.5
        mask_down = np.maximum(-ones, np.minimum(ones, down_dist))*0.5+0.5
        mask_up = np.maximum(-ones, np.minimum(ones, up_dist))*0.5+0.5
        mask = mask_left*mask_right*mask_down*mask_up
        cut = np.asarray(mask_down*mask_up>0).nonzero()

        mask_z = self.z * mask
        mask_z_targ = self.z_targ * mask
        mask_z_bckg = self.z_bckg * mask

        cut_shape = (int(len(cut[0])/mask_z.shape[1]), mask_z.shape[1])
        cut_z = mask_z[cut].reshape(cut_shape)
        cut_z_targ = mask_z_targ[cut].reshape(cut_shape)
        cut_z_bckg = mask_z_bckg[cut].reshape(cut_shape)
    
        flux = np.sum(mask_z)
        flux_targ = np.sum(mask_z_targ)
        flux_bckg = np.sum(mask_z_bckg)
        trace = np.sum(cut_z, axis=1)
        trace_targ = np.sum(cut_z_targ, axis=1)
        trace_bckg = np.sum(cut_z_bckg, axis=1)

        return flux, flux_targ, flux_bckg, trace, trace_targ, trace_bckg

    
    def add_slices(self, n=slice_n, cen=None):

        width = self.slice_width
        length = self.ysize
        if cen == None:
            cen = (0,0)
        hwidth = width.value / 2
        hlength = length.value / 2
        shift = ((n+1)*width.value)/2
        cens = np.arange(cen[0]-shift, cen[0]+shift, width.value)
        self.flux_slice = 0.
        self.flux_targ_slice = 0.
        self.flux_bckg_slice = 0.
        for i, c in enumerate(cens[1:]):
            print("Designing slice %i on field..." % i, end='\r')

            flux, flux_targ, flux_bckg, trace, trace_targ, trace_bckg = self.add_slice((c, cen[1]))
            self.flux_slice += flux
            self.flux_targ_slice += flux_targ
            self.flux_bckg_slice += flux_bckg
            if i == 0:
                self.fluxes = np.array([flux.value])
                self.fluxes_targ = np.array([flux_targ.value])
                self.fluxes_bckg = np.array([flux_bckg.value])
                self.traces = [trace]
                self.traces_targ = [trace_targ]
                self.traces_bckg = [trace_bckg]
            else:
                self.fluxes = np.append(self.fluxes, flux.value)
                self.fluxes_targ = np.append(self.fluxes_targ, flux_targ.value)
                self.fluxes_bckg = np.append(self.fluxes_bckg, flux_bckg.value)
                self.traces = np.vstack((self.traces, trace.value))
                self.traces_targ = np.vstack((self.traces_targ, trace_targ.value))
                self.traces_bckg = np.vstack((self.traces_bckg, trace_bckg.value))
        print("Slices designed on field.     ")
    
        self.slice_area = min((n * width-self.pix_xsize*au.pixel) * (length-self.pix_ysize*au.pixel), 
                              (n * width-self.pix_xsize*au.pixel) * (self.ysize-self.pix_ysize*au.pixel))
        self.slice_area = min(n*width*length, n*width*self.ysize)
        self.bckg_slice = self.flux_bckg_slice
        self.targ_slice = self.flux_slice-self.bckg_slice
        self.spec.targ_slice = self.targ_slice
        self.losses = 1-(self.flux_targ_slice.value)/np.sum(self.z_targ.value)

        # Update spectrum with flux into slices and background
        if self.func == 'gaussian':#or 1==0:
            self.spec.targ_conv = gaussian_filter(
                self.spec.targ_ext, self.sigma.value)*self.spec.targ_ext.unit
            self.spec.bckg_conv = gaussian_filter(
                self.spec.bckg_ext, self.sigma.value)*self.spec.bckg_ext.unit

        else:
            self.spec.targ_conv = self.spec.targ_ext#*(1-self.losses)
            self.spec.bckg_conv = self.spec.bckg_ext
        
            
    def draw(self):
        fig_p, self.ax_p = plt.subplots(figsize=(10,7))
        self.ax_p.set_title("Photon balance (slit)")
        #sl_v = np.array([self.spec.targ_tot.value, np.sum(self.targ_slice.value), 
        #                 np.sum(self.z_targ.value)-np.sum(self.targ_slice.value)])
        sl_v = np.array([np.sum(self.z_targ.value), np.sum(self.targ_slice.value), 
                         np.sum(self.z_targ.value)-np.sum(self.targ_slice.value)])
        sl_l = ["total: %2.3e %s"  % (sl_v[0], self.spec.targ_tot.unit), 
                "on slit: %2.3e %s"  % (sl_v[1], self.z_targ.unit), 
                "off slit: %2.3e %s"  % (sl_v[2], self.z_targ.unit)] 
        sl_c = ['C0', 'C0', 'C0']
        p0 = self.ax_p.pie([sl_v[0]], colors=[sl_c[0]], startangle=90, radius=1,
                      wedgeprops=dict(width=0.2, edgecolor='w'))
        p1 = self.ax_p.pie(sl_v[1:], colors=sl_c[1:], autopct='%1.1f%%', startangle=90, radius=0.8,
                          wedgeprops=dict(width=0.2, edgecolor='w'))
        p1[0][0].set_alpha(1/2)
        p1[0][1].set_alpha(1/6)
        self.ax_p.legend([p0[0][0],p1[0][0]], [sl_l[0]]+sl_l[1::2])

        fig, self.ax = plt.subplots(figsize=(8,8))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.contourf(self.x, self.y, self.z, 100, vmin=0)
        for rect in self.rects:
            self.ax.add_patch(rect)
        self.ax.set_title('PSF')
        self.ax.set_xlabel('X (%s)' % self.xsize.unit)
        self.ax.set_ylabel('Y (%s)' % self.xsize.unit)
        cax.xaxis.set_label_position('top')
        cax.set_xlabel('Photon')
        fig.colorbar(im, cax=cax, orientation='vertical')

    
    def invrad(self):
        ret = 1/np.sqrt(self.x**2+self.y**2)
        sel = self.x**2+self.y**2 < targ_invrad_params['r_eff']**2
        return ret, sel
        
    def gaussian(self, cen=psf_cen):
        ampl = 1
        theta = 0
        sigma = self.seeing.value/2 / np.sqrt(2*np.log(2))
        m = Gaussian2D(ampl, cen[0], cen[1], sigma, sigma, theta)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], sigma, sigma,
                            theta)
        self.sigma = m.x_stddev
        self.fwhm = m.x_fwhm


    def moffat(self, cen=psf_cen):
        ampl = 1
        alpha = 3
        gamma = self.seeing.value/2 / np.sqrt(2**(1/alpha)-1)
        m = Moffat2D(1, cen[0], cen[1], gamma, alpha)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], gamma,
                            alpha)
        self.fwhm = m.fwhm


    def sersic(self):
        m = Sersic2D(**targ_sersic_params)
        sel = self.x**2+self.y**2 < targ_sersic_params['r_eff']**2
        return m.evaluate(self.x, self.y, **targ_sersic_params), sel
        
        
    def tophat(self, cen=(0,0)):
        self.z = np.array((self.x-cen[0])**2 + (self.y-cen[1])**2
                          < (self.seeing.value/2)**2, dtype=int)
        self.fwhm = self.seeing.value


class Spec(object):

    def __init__(self, phot, file=None, wmin=wmin, wmax=wmax,
                 dw=1e-3*au.nm, templ=spec_templ):
        self.phot = phot
        self.file = file
        self.wmin = wmin
        self.wmax = wmax
        self.wmean = 0.5*(wmin+wmax)
        self.wave = np.arange(wmin.value, wmax.value, dw.value)*wmin.unit

        try:
            flux_bckg = self.skycalc()
            skycalc = True
        except:
            skycalc = False
                
        # Extrapolate extinction
        spl = cspline(self.phot.atmo_wave, self.phot.atmo_ex)(self.wave)
        self.atmo_ex = pow(10, -0.4*spl*airmass)

        flux_targ = getattr(self, templ)()
  
        mag_u = self.create(flux_targ, 'targ')
        print("Input spectra created; target magnitude in the U band: %3.2f." % mag_u)

        if skycalc:
            self.create(flux_bckg, 'bckg', norm_flux=False)
            print("Sky spectrum imported from SkyCalc_input_NEW_Out.fits.")
            
        else:
            self.create(np.ones(self.wave.shape), 'bckg')
            print("Flat sky model created.")

        
    def create(self, flux, obj='targ', norm_flux=True):
        if norm_flux:
            raw = flux * getattr(self.phot, obj)
        else:
            raw = flux * au.ph/au.nm/au.arcsec**2
        
        ext = raw*self.atmo_ex
        tot = np.sum(ext)/len(ext) * (self.wmax-self.wmin)
        setattr(self, obj+'_raw', raw)
        setattr(self, obj+'_ext', ext)
        setattr(self, obj+'_tot', tot)
        
        u = np.where(np.logical_and(self.wave > np.min(self.phot.wave_u), self.wave < np.max(self.phot.wave_u)))
        waveu = self.wave[u]
        dwaveu = np.median(self.wave[1:]-self.wave[:-1])
        fluxu = self.targ_raw[u]
        spl_u = cspline(self.phot.wave_u, self.phot.flux_u)(waveu) 
    
        flux_norm = np.sum(fluxu*spl_u)/np.sum(spl_u)
        mag_u = -2.5*np.log10(flux_norm / (flux_ref_Vega['U'] * self.phot.area * texp))
            
        return mag_u

    def custom(self):
        name = self.file    
        try:
            data = Table(ascii.read(name, names=['col1', 'col2', 'col3', 'col4']))
            wavef = data['col1']*0.1 * au.nm
            fluxf = data['col2']
            
        except:
            data = Table(ascii.read(name, names=['col1', 'col2'], format='no_header'), dtype=(float, float))
            wavef = data['col1']*0.1 * au.nm
            fluxf = data['col2']
        if zem != None:
            wavef = wavef*(1+zem)

        if igm_abs in ['simple', 'inoue'] and zem != None:            
            fluxf = getattr(self, igm_abs+'_abs')(wavef, fluxf)
                        
        band = np.where(np.logical_and(wavef>np.min(self.phot.wave_band), wavef<np.max(self.phot.wave_band)))
        waveb = wavef[band]
        dwaveb = np.median(wavef[1:]-wavef[:-1])
        fluxb = fluxf[band]
        spl_band = cspline(self.phot.wave_band, self.phot.flux_band)(waveb) 

        self.wavef = wavef
        self.fluxf = fluxf

        spl = cspline(wavef, fluxf)(self.wave.value)
        flux = spl/np.sum((spl_band*fluxb*dwaveb).value)

        return flux #* self.atmo_ex

    
    def draw_in(self, bckg=True, show=True):    

        if bckg:
            fig_p, self.ax_p = plt.subplots(figsize=(10,7))
            self.ax_p.set_title("Photon balance (telescope)")
            sl_v = [self.targ_tot.value, self.bckg_tot.value]
            sl_l = ["target: %2.3e %s"  % (sl_v[0], self.targ_tot.unit), 
                    "background: %2.3e %s" % (sl_v[1], self.bckg_tot.unit)]
            sl_c = ['C0', 'C1']
            #print(sl_v)
            p = self.ax_p.pie(sl_v, colors=sl_c, autopct='%1.1f%%', startangle=90, 
                          wedgeprops=dict(width=0.2, edgecolor='w'))
            self.ax_p.legend(p[0], sl_l)

        fig, self.ax = plt.subplots(figsize=(10,5))
        self.ax.set_title("Spectrum")
        self.ax.plot(self.wave, self.targ_raw, label='Target raw')
        self.ax.plot(self.wave, self.targ_ext, label='Target extincted', linestyle='--', c='C0')
        if bckg: self.ax.plot(self.wave, self.bckg_raw, label='Background (per arcsec2)')
        
        self.ax.set_xlabel('Wavelength (%s)' % self.wave.unit)
        self.ax.set_ylabel('Flux density\n(%s)' % self.targ_raw.unit)
        self.ax.grid(linestyle=':')
            
        if show: 
            self.ax.legend(loc=2, fontsize=8)
            plt.show()


        
    def draw(self):
        fig_p, self.ax_p = plt.subplots(figsize=(7,7))
        self.ax_p.set_title("Photon balance (CCD)")
        sl_v = np.array([self.targ_tot.value, np.sum(self.targ_slice.value), 
                         np.sum(self.z_targ.value)-np.sum(self.targ_slice.value),
                         self.targ_sum.value, np.sum(self.z_targ.value)-self.targ_sum.value,
                         self.flux_final_tot.value, np.sum(self.z_targ.value)-self.flux_final_tot.value])
        sl_l = ["target: %2.3e %s"  % (sl_v[0], self.targ_tot.unit), 
                "on slit: %2.3e %s"  % (sl_v[1], self.z_targ.unit), 
                "off slit: %2.3e %s"  % (sl_v[2], self.z_targ.unit),
                "on CCD: %2.3e %s"  % (sl_v[3], self.targ_sum.unit), 
                "off CCD: %2.3e %s"  % (sl_v[4], self.targ_sum.unit),
                "extracted: %2.3e %s"  % (sl_v[5], self.flux_final_tot.unit), 
                "missed: %2.3e %s"  % (sl_v[6], self.flux_final_tot.unit)] 
        sl_c = ['C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0']
        p0 = self.ax_p.pie([sl_v[0]], colors=[sl_c[0]], startangle=90, radius=1,
                      wedgeprops=dict(width=0.2, edgecolor='w'))
        p1 = self.ax_p.pie(sl_v[1:3], colors=sl_c[1:3], startangle=90, radius=0.8,
                           wedgeprops=dict(width=0.2, edgecolor='w'))
        p2 = self.ax_p.pie(sl_v[3:5], colors=sl_c[3:5], startangle=90, radius=0.6, 
                           wedgeprops=dict(width=0.2, edgecolor='w'))
        p3 = self.ax_p.pie(sl_v[5:], colors=sl_c[5:], autopct='%1.1f%%', startangle=90, radius=0.4, 
                           wedgeprops=dict(width=0.2, edgecolor='w'))
        p1[0][0].set_alpha(3/4)
        p1[0][1].set_alpha(1/6)
        p2[0][0].set_alpha(2/4)
        p2[0][1].set_alpha(1/6)
        p3[0][0].set_alpha(1/4)
        p3[0][1].set_alpha(1/6)
        self.ax_p.legend([p0[0][0],p1[0][0],p2[0][0],p3[0][0]], [sl_l[0]]+sl_l[1::2])

        self.draw_in(bckg=False, show=False)
        self.ax.set_title("Spectrum")
        
        for a in range(arm_n):
            line1, = self.ax.plot(self.arm_wave[a], self.arm_targ[a] \
                * self.tot_eff(self.arm_wave[a], self.m_d[a], self.M_d[a]), c='C3')
            if arm_n > 1:
                line2 = self.ax.scatter(self.wave_extr[a,:], self.flux_extr[a,:], s=2, c='C0')
                line3 = self.ax.scatter(self.wave_extr[a,:], self.err_extr[a,:], s=2, c='C1')
            else:
                line2 = self.ax.scatter(self.wave_extr[:], self.flux_extr[:], s=2, c='C0')
                line3 = self.ax.scatter(self.wave_extr[:], self.err_extr[:], s=2, c='C1')
                
        line1.set_label('On detector')            
        line2.set_label('Extracted')
        line3.set_label('Extracted (error)')

        self.ax.legend(loc=2, fontsize=8)

        fig_noise, self.ax_noise = plt.subplots(figsize=(10,5))
        self.ax_noise.set_title("Noise spectrum")
        for a in range(arm_n):
            if arm_n > 1:
                line2 = self.ax_noise.scatter(self.wave_extr[a,:], self.err_targ_extr[a,:], s=2, c='C0')
                line3 = self.ax_noise.scatter(self.wave_extr[a,:], self.err_bckg_extr[a,:], s=2, c='C1')
                line4 = self.ax_noise.scatter(self.wave_extr[a,:], self.err_dark_extr[a,:], s=2, c='C2')
                line5 = self.ax_noise.scatter(self.wave_extr[a,:], self.err_ron_extr[a,:], s=2, c='C3')
            else:
                line2 = self.ax_noise.scatter(self.wave_extr[:], self.err_targ_extr[:], s=2, c='C0')
                line3 = self.ax_noise.scatter(self.wave_extr[:], self.err_bckg_extr[:], s=2, c='C1')
                line4 = self.ax_noise.scatter(self.wave_extr[:], self.err_dark_extr[:], s=2, c='C2')
                line5 = self.ax_noise.scatter(self.wave_extr[:], self.err_ron_extr[:], s=2, c='C3')

        line2.set_label('Target')
        line3.set_label('Background')
        line4.set_label('Dark')
        line5.set_label('RON ')
        self.ax_noise.legend(loc=2, fontsize=8)
        self.ax_noise.set_xlabel('Wavelength')
        self.ax_noise.set_ylabel('Flux\n(ph/extracted pix)')
        
        fig_pix, self.ax_pix = plt.subplots(figsize=(10,5))
        self.ax_pix.set_title("Extraction spectrum")
        for a in range(arm_n):
            if arm_n > 1:
                line1 = self.ax_pix.scatter(self.wave_extr[a,:], self.pix_extr[a,:], s=2, c='red')
            else:
                line1 = self.ax_pix.scatter(self.wave_extr[:], self.pix_extr[:], s=2, c='red')
        self.ax_pix.set_xlabel('Wavelength')
        self.ax_pix.set_ylabel('(pix/extracted pix)')
        
        fig_snr, self.ax_snr = plt.subplots(figsize=(10,5))
        self.ax_snr.set_title("SNR")
        linet, = self.ax_snr.plot(self.wave_snr, self.snr, linestyle='--', c='black')
        linet.set_label('SNR')
        self.ax_snr.text(0.99, 0.92,
                              "Median SNR: %2.1f" % np.median(self.snr),
                              ha='right', va='top',
                              transform=self.ax_snr.transAxes)
        self.ax_snr.legend(loc=2, fontsize=8)

        self.ax_snr.set_xlabel('Wavelength')
        self.ax_snr.set_ylabel('SNR (1/extracted pix)')
        self.ax_snr.grid(linestyle=':')
        
    def flat(self):
        return np.ones(self.wave.shape) #* self.atmo_ex
    
    def inoue_abs(self, wave, flux):
        inoue = ascii.read('../database/Table2_inoue.dat')
        
        w = wave.value
        w_ll = 91.18
        
        tau_ls_laf = np.zeros(len(wave))
        tau_ls_dla = np.zeros(len(wave))
        tau_lc_laf = np.zeros(len(wave))
        tau_lc_dla = np.zeros(len(wave))

        for (i, wi, a_laf_1, a_laf_2, a_laf_3, a_dla_1, a_dla_2) in inoue:
            wi = 0.1*wi

            # Eq. 21
            s_1 = np.where(np.logical_and(w>wi, np.logical_and(w<wi*2.2, w<wi*(1+zem))))[0]
            s_2 = np.where(np.logical_and(w>wi, np.logical_and(np.logical_and(w>wi*2.2, w<wi*5.7), w<wi*(1+zem))))[0]
            s_3 = np.where(np.logical_and(w>wi, np.logical_and(np.logical_and(w>wi*2.2, w>wi*5.7), w<wi*(1+zem))))[0]
            tau_ls_laf[s_1] = tau_ls_laf[s_1] + a_laf_1 * (w[s_1]/wi)**1.2
            tau_ls_laf[s_2] = tau_ls_laf[s_2] + a_laf_2 * (w[s_2]/wi)**3.7
            tau_ls_laf[s_3] = tau_ls_laf[s_3] + a_laf_3 * (w[s_3]/wi)**5.5

            # Eq. 22
            s_4 = np.where(np.logical_and(w>wi, np.logical_and(w<wi*3.0, w<wi*(1+zem))))[0]    
            s_5 = np.where(np.logical_and(w>wi, np.logical_and(w>wi*3.0, w<wi*(1+zem))))[0]    
            tau_ls_dla[s_4] = tau_ls_dla[s_4] + a_dla_1 * (w[s_4]/wi)**2.
            tau_ls_dla[s_5] = tau_ls_dla[s_5] + a_dla_2 * (w[s_5]/wi)**3.
        
        # Eq. 25
        if zem < 1.2:  
            s_6 = np.where(np.logical_and(w>w_ll, w<w_ll*(1+zem)))[0]
            tau_lc_laf[s_6] = 0.325*((w[s_6]/w_ll)**1.2 - (1+zem)**-0.9 * (w[s_6]/w_ll)**2.1)

        # Eq. 26
        elif zem < 4.7:
            s_7 = np.where(np.logical_and(w>w_ll, np.logical_and(w<2.2*w_ll, w<w_ll*(1+zem))))[0]
            s_8 = np.where(np.logical_and(w>w_ll, np.logical_and(w>2.2*w_ll, w<w_ll*(1+zem))))[0]
            tau_lc_laf[s_7] = 2.55e-2*(1+zem)**1.6 * (w[s_7]/w_ll)**2.1 + 0.325*(w[s_7]/w_ll)**1.2 - 0.250*(w[s_7]/w_ll)**2.1 
            tau_lc_laf[s_8] = 2.55e-2*((1+zem)**1.6 * (w[s_8]/w_ll)**2.1 - (w[s_8]/w_ll)**3.7)

        # Eq. 27
        else: 
            s_9 = np.where(np.logical_and(w>w_ll, np.logical_and(w<2.2*w_ll, w<w_ll*(1+zem))))[0]
            s_a = np.where(np.logical_and(w>w_ll, np.logical_and(w>2.2*w_ll, np.logical_and(w<5.7*w_ll, w<w_ll*(1+zem)))))[0]
            s_b = np.where(np.logical_and(w>w_ll, np.logical_and(w>2.2*w_ll, np.logical_and(w>5.7*w_ll, w<w_ll*(1+zem)))))[0]
            tau_lc_laf[s_9] = 5.22e-4*(1+zem)**3.4 * (w[s_9]/w_ll)**2.1 + 0.325*(w[s_9]/w_ll)**1.2 - 3.14e-2*(w[s_9]/w_ll)**2.1
            tau_lc_laf[s_a] = 5.22e-4*(1+zem)**3.4 * (w[s_a]/w_ll)**2.1 + 0.218*(w[s_a]/w_ll)**2.1 - 2.55e-2*(w[s_a]/w_ll)**3.7
            tau_lc_laf[s_b] = 5.22e-2*((1+zem)**3.4 * (w[s_b]/w_ll)**2.1 - (w[s_b]/w_ll)**5.5)

        # Eq. 28
        if zem < 2.0:
            s_c = np.where(np.logical_and(w>w_ll, w<w_ll*(1+zem)))[0]
            tau_lc_dla[s_c] = 0.211*(1+zem)**2. - 7.66e-2*(1+zem)**2.3 * (w[s_c]/w_ll)**(-0.3) - 0.135*(w[s_c]/w_ll)**2.
                           
        # Eq. 29
        else:               
            s_d = np.where(np.logical_and(w>w_ll, np.logical_and(w<3.0**w_ll, w<w_ll*(1+zem))))[0]
            s_e = np.where(np.logical_and(w>w_ll, np.logical_and(w>3.0*w_ll, w<w_ll*(1+zem))))[0]
            tau_lc_dla[s_d] = 0.634 + 4.7e-2*(1+zem)**3. - 1.78e-2*(1+zem)**3.3 * (w[s_d]/w_ll)**(-0.3) \
                              - 0.135*(w[s_d]/w_ll)**2. - 0.291*(w[s_d]/w_ll)**(-0.3)

            tau_lc_dla[s_e] = 4.7e-2*(1+zem)**3. - 1.78e-2*(1+zem)**3.3 * (w[s_e]/w_ll)**(-0.3) - 2.92e-2*(w[s_e]/w_ll)**3.

        tau = tau_ls_laf + tau_ls_dla + tau_lc_laf + tau_lc_dla
        corr = np.ones(len(flux))
        z = wave.value/121.567 - 1
        corr[z<zem] = np.exp(tau[z<zem])

        return flux/corr


    def PL(self, index=-1.5):
        return (self.wave.value/self.phot.wave_ref.value)**index * self.atmo_ex

    
    def qso(self):
        if self.file is None:
            name = qso_file
        else:
            name = self.file
        try:
            data = fits.open(name)[1].data
            data = data[:-1]
            data = data[np.logical_and(data['wave']*0.1 > self.wmin.value,
                                       data['wave']*0.1 < self.wmax.value)]
            wavef = data['wave']*0.1 * au.nm
            fluxf = data['flux']
            spl = cspline(wavef, fluxf)(self.wave.value)
            sel = np.where(self.wave.value<313)
            spl[sel] = 1.0
        except:
            pass
            
        flux = spl#/np.mean(spl) #* au.photon/au.nm
        #if igm_abs and zem != None:
        #    flux = self.lya_abs(flux)
        return flux * self.atmo_ex
    
    
    def simple_abs(self, wave, flux):
        logN_0 = 12
        logN_1 = 14
        logN_2 = 18
        logN_3 = 19
        index = -1.65

        f_0 = 1.0
        f_1 = 1e14/np.log(1e14)
        f_2 = np.log(1e18)/np.log(1e14)*1e5

        tau_norm = 0.0028
        tau_index = 3.45
        qso_zprox = zem - (1.0 + zem) * 10000 * au.km/au.s / ac.c
        
        frac = 1
        
        corr = np.ones(len(flux))
        z = wave.value/121.567 - 1

        corr[z<zem] = (1-np.exp(tau_norm*(1+qso_zprox)**tau_index*frac)*f_0) \
                           / (zem-qso_zprox) * (z[z<zem]-qso_zprox.value) \
                           + np.exp(tau_norm*(1+qso_zprox)**tau_index*frac)*f_0
        corr[z<qso_zprox] = np.exp(tau_norm*(1+z[z<qso_zprox])**tau_index*frac)*f_0
        return flux / corr


    def skycalc(self):
        name = 'SkyCalc_input_NEW_Out.fits'   
        data = Table.read(name)
        wavef = data['lam'] * au.nm
        fluxf = data['flux'] * self.phot.area.to(au.m**2).value * self.phot.texp * 1e-3
        atmo_trans = data['trans']
        self.phot.atmo_wave = wavef
        self.phot.atmo_ex = (1-atmo_trans)
        spl = cspline(wavef, fluxf)(self.wave.value)
        flux = spl #* au.photon/au.nm
        return flux

    
    def star(self):
        if self.file is None:
            name = star_file
        else:
            name = self.file
        data = Table(ascii.read(name, data_start=2, names=['col1', 'col2'], format='no_header')[2:], dtype=(float, float))#name)
        wavef = data['col1']*0.1 * au.nm
        fluxf = data['col2']
        spl = cspline(wavef, fluxf)(self.wave.value)
        flux = spl/np.mean(spl) #* au.photon/au.nm
        return flux * self.atmo_ex

    
class Sim():
    
    def __init__(self):
        pass
    

    def check(self, start=True, *args):
        refresh = False
        if start: 
            for o in args:
                if hasattr(self, '_'+o) and o+'_pars' in globals():
                    for k in globals()[o+'_pars']:
                        try:
                            r = self.__dict__[k]!=globals()[k]
                            try:
                                if len(r.shape)>1: r = np.ravel(r)
                            except:
                                pass
                            try:
                                if r.size > 0: r = any(r)
                            except:
                                pass
                        except:
                            r = False
                        if r:
                            refresh = True
                    if refresh:
                        delattr(self, '_'+o)
            self.start()
        del refresh
        
        for o in args:
            if not hasattr(self, '_'+o):
                getattr(self, o+'_create')()
                
    
    def ccd(self):
        self.ccd_create()
        self.ccd_draw()


    def ccd_create(self):            
        self.check(True, 'phot', 'spec', 'psf')
        self._ccd = CCD(self._psf, self._spec, xsize=ccd_xsize, ysize=ccd_ysize, xbin=ccd_xbin, ybin=ccd_ybin, func=extr_func)
        self._ccd.add_arms(n=arm_n, wave_d=wave_d, wave_d_shift=wave_d_shift)

        
    def ccd_draw(self):
        self.check(False, 'phot', 'spec', 'psf', 'ccd')
        self._ccd.draw()
        plt.show()


    def phot_create(self):
        self.start()
        self._phot = Photons(targ_mag=targ_mag, bckg_mag=bckg_mag, texp=texp)
        
        
    def psf(self):
        self.psf_create()
        self.psf_draw()
        
        
    def psf_create(self):
        self.check(True, 'phot', 'spec')
        self._psf = PSF(self._spec, seeing=seeing, slice_width=slice_width, xsize=slice_length,
                        ysize=slice_length, func=psf_func)
        self._psf.add_slices(n=slice_n)


    def psf_draw(self):
        self.check(False, 'phot', 'spec', 'psf')
        self._psf.draw()        
        plt.show()

        
    def spec_in(self):
        self.spec_create()
        self._spec.draw_in()
        

    def spec_create(self):
        self.check(True, 'phot')
        self._spec = Spec(self._phot, file=spec_file, wmin=wmin, wmax=wmax, templ=spec_templ)


    def spec_draw(self):
        self.check(True, 'phot', 'spec', 'psf', 'ccd')
        self._ccd.extr_arms(n=arm_n, slice_n=slice_n)
        self._spec.draw()

        
    def spec_save(self, file):
        wmin, wmax = np.min(self._spec.wave), np.max(self._spec.wave)
        w = np.where(np.logical_and(self._spec.wavef.value > wmin.value, self._spec.wavef.value < wmax.value))
        wavef = self._spec.wavef[w].to(au.Angstrom)
        fluxf = cspline(self._spec.wave.to(au.Angstrom).value, 
                        (self._spec.targ_raw.to(au.ph/au.Angstrom)/self._phot.texp/self._phot.area).value)(wavef.value)
        t = Table([wavef, fluxf], 
                  names=['wave','flux'])
        comment = "l(A) photons/cm2/s/A, z = %3.4f\n%3.4f Vega_Vmag" % (zem, self._phot.targ_mag)
        t.meta['comments'] = [comment]
        t.write(file, format='ascii.no_header', formats={'wave': '%2.4f', 'flux': '%2.12e'}, 
                overwrite=True)  
       
        
    def start(self):
        for k in self.__dict__:
            globals()[k] = self.__dict__[k]