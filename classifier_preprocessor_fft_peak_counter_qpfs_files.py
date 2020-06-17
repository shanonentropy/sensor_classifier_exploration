# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:38:27 2019

@author: zahmed

this program is part of the SENSOR_CLASSIFIER program's pre-processing routine
it will take in all the data from the sensor folder and display it 
"""
import os
import pandas as pd
from sklearn.preprocessing import minmax_scale
#from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.interpolate import splrep, sproot
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot

#scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\raw_data\qps_fbg_classification\cycle_data'

 
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
Q = []
asym = []
number_of_peaks = []
fbg_decomp = []

cols = ['x', 'y', 'z']
for  fname in os.listdir(path_dir):
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 6,  engine = 'python',names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
    df['y_invert'] = df['y'].mean()-df.y
    plt.plot(df.x, df.y)
#    plt.show()
    base = peakutils.baseline(df.y)
    indexes= peakutils.indexes(df.y-base, thres=.35, min_dist=200)
#    pplot(df.x,df.y, indexes); plt.show()
    print(indexes)
    for i in indexes:
        if i[(i<600) & (i>350)]:
            peak_x=i
            a = peak_x+30
            b = peak_x -150
            x2,y2=df.x[b:a],df.y_invert[b:a]
#            x2 , y2=df.x , df.y_invert
#            plt.plot(x2 , y2); plt.show()
            base = peakutils.baseline(y2, 2)
#            plt.figure(figsize=(10,6))
#            plt.plot(x2, y2-base)
            skewness = stats.skew(df.y)
            tck = interpolate.splrep(x2,y2,s=.001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
#            tck = interpolate.splrep(x2,y2,s=.01) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
            x_ = np.arange (np.min(x2),np.max(x2), 0.003)
            y_ = interpolate.splev(x_, tck, der=0)
            plt.plot(x2,y2)
            plt.scatter(x_,y_)
#            plt.show()
            HM = (np.max(y_)-np.min(y_))/2
            w = splrep(x_, y_ - HM)
            w = [1549.5, 1549.6]
            try:
                if len(sproot(w))%2==0:
                    r1 , r2 = sproot(w)
                    print(r1 , r2)
                    FWHM = np.abs(r1 - r2)
#                    half_width.append(FWHM)
                    center_wavelength = r1 + FWHM/2
                    Q.append(center_wavelength/FWHM)
#                    peak_center.append(center_wavelength)
#                    filenames.append(fname)
                    df['x_scale'] = minmax_scale(df.x, feature_range=(0,1))
                    df['y_scale'] = minmax_scale(df.y, feature_range=(0,1))
                    freq_axis = np.fft.fftfreq(df['x_scale'].count())
                    power = np.fft.fft(df['y_scale']).real
#                    trunc = int(len(freq_axis)/2)
                    power_freq = pd.DataFrame(power[:180], columns=['power'])
                    plt.plot(freq_axis[:180], power_freq)
                    plt.show()
                    fft_profile = power_freq.transpose()
                    fft_profile['Q'] = 20667
                    fft_profile['device'] = 'qpfs'
                    fft_profile['asym'] = skewness
                    fft_profile['num_peaks'] = 1
#                    fft_profile.to_csv('fft'+fname)          
            except (TypeError, ValueError):
                print(fname,'error')
                continue
            
#    df1 = pd.DataFrame(x_)
###    print(df1.head(3))
#    df1['x_scale'] = minmax_scale(x_, feature_range=(0,1))
#    df1['y_scale'] = minmax_scale(y_, feature_range=(0,1))
#    freq_axis = np.fft.fftfreq(df1['x_scale'].count())
#    power = np.fft.fft(df1['y_scale']).real
#    trunc = int(len(freq_axis)/2)
#    power_freq = pd.DataFrame(power[:50], columns=['power'])
#    plt.plot(freq_axis[:50], power_freq)
#    plt.show()
##    plt.xlim(40,180 )
##    plt.ylim(-.1,1)
#    pad = np.arange(51,181,1)
#    power_freq_pad = pd.concat([power_freq, pd.DataFrame(index = pad)], axis = 0, sort=False)
#    power_freq_pad.fillna(0,inplace = True)
##    plt.plot(power_freq)
#    fft_profile = power_freq.transpose()
#    fft_profile['Q'] = center_wavelength/FWHM
#    fft_profile['device'] = 'ring_resonator_single_mode'
#    fft_profile['asym'] = skewness
#    fft_profile['num_peaks'] = len(indexes)
##    fft_decomp.append(fft_profile)
#    fft_profile.to_csv('fft'+fname)
#    
    

    
#df_q = pd.DataFrame({'filnames':file_names,'fft':fft_p ,'quality_factor':Q, 'skew':asym, 'number_of_peaks': number_of_peaks})
#df_q.to_csv('peak_characteristics')