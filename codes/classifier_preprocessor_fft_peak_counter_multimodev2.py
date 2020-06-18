# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:38:27 2019

@author: zahmed

this program is part of the SENSOR_CLASSIFIER program's pre-processing routine
it will take in all the data from the sensor folder and display it 
"""
import os
import pandas as pd
#from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.interpolate import splrep, sproot
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import peakutils
#from peakutils.plot import plot as pplot

scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\raw_data\ring_resonator_multiple_modes\CHIP_3_temp_dep\set2'


#loop over the files and then create a list of file names to later iterate over
    
''' for each spectra we need to extract the following set of information
number of peaks 
    if more than one peak, peak-to-peak distace (ppd) and delta ppd
Q of the device
from the normalized spectra, skewness
and intensity of profile of the spectra

the first part is to just feed in data with profile and label and 
see if the classifier works, if not, keep adding more features

so this program will just take in the data, fit it, create a dataset with 
known pitch of 0.003 nm and output data ninmax scaled profile data with the 
same name

'''
filenames = []
Q1 = []
asym = []
number_of_peaks = []
fbg_decomp = []

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
#    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 4,  engine = 'python', names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
    indexes = peakutils.indexes(df.y, thres=0.2, min_dist=100)
    freq_axis = np.fft.fftfreq(df['x'].count())
    power = np.fft.fft(df['y']).real
    df1 = pd.DataFrame(power)
    power_scale = scaler.fit_transform(df1)
    plt.plot(power_scale[:180])
    power_freq = pd.DataFrame(power_scale[:180], columns=['power'])
#    plt.plot(power_freq)
    fft_profile = power_freq.transpose()
    fft_profile.reset_index(drop=True, inplace=True)
    fft_profile['device'] = 'ring_resonator_multimode'
    fft_profile['asym'] = stats.skew(df.y)
    fft_profile['num_peaks'] = len(indexes)
#    df.plot('x','y') 
#    pplot(df.x, df.y, indexes)
#    plt.show()
#    print(indexes[-1])
    peak = indexes[-1]
    left_o_peak = peak - 400
    right_o_peak = peak + 400
    tck = interpolate.splrep(df.x[left_o_peak:right_o_peak],df.y[left_o_peak:right_o_peak],s=0.000001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
    x_ = np.arange (df.x[left_o_peak:right_o_peak].min(),df.x[left_o_peak:right_o_peak].max(), 0.003)
    y_ = interpolate.splev(x_, tck, der=0)
#    plt.plot(x_,y_, 'ro')
#    plt.plot(df.x[left_o_peak:right_o_peak], df.y[left_o_peak:right_o_peak])
#    plt.show()
    HM =(np.max(y_)-np.min(y_))/2
    w = splrep(x_, y_ - HM)
#    print(sproot(w))
    try:
        if len(sproot(w))%2 == 0:
            r1 , r2 = sproot(w)
#            print(r1, r2)
            FWHM = np.abs(r1 - r2)
            center_wavelength = r1 + FWHM/2
            Q = center_wavelength/FWHM
#            print(Q)
    except (TypeError, ValueError):
#        print(fname,'error')
        Q = 11346
        continue
    
    fft_profile['Q'] = Q
    print(fft_profile)
#    Q1.append(Q)
#    fft_decomp.append(fft_profile)
    fft_profile.to_csv('fft'+fname)
    
    

    
#df_q = pd.DataFrame({'filnames':file_names,'fft':fft_p ,'quality_factor':Q, 'skew':asym, 'number_of_peaks': number_of_peaks})
#df_q.to_csv('peak_characteristics')
