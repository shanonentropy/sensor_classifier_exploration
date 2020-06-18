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
from peakutils.plot import plot as pplot

scaler =StandardScaler()

#path to directory with the relevant files
path_dir = r'C:\Interpolation_Project\classification\raw_data\ring_resonator_classification\all_pass'


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

cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 4,  engine = 'python', names =cols )
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
    df['y_invert'] = df['y'].mean()-df.y
#    df.plot('x','y_invert')  
    base = peakutils.baseline(df.y_invert, 2)
    indexes= peakutils.indexes(df.y_invert-base, thres=0.00001, min_dist=200)
#    print(indexes)
#    pplot(df.x,df.y_invert, indexes)
    for i in indexes:
        if i[(i<100) & (i>5)]:
            peak_x=i
            x2 , y2=df.x , df.y_invert
            skewness = stats.skew(y2)
            tck = interpolate.splrep(x2,y2,s=.00000001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
            x_ = np.arange (np.min(x2),np.max(x2), 0.003)
            y_ = interpolate.splev(x_, tck, der=0)

            HM = (np.max(y_)-np.min(y_))/2
            w = splrep(x_, y_ - HM, k=3)
#            print(sproot(w_j))
            try:
                if len(sproot(w))%2==0:
                    r1 , r2 = sproot(w)
#                    print(r1 , r2)
                    FWHM = np.abs(r1 - r2)
#                    half_width.append(FWHM)
                    center_wavelength = r1 + FWHM/2
#                    peak_center.append(center_wavelength)
            except (TypeError, ValueError):
#                print(fname,'error')
                continue
    df1 = pd.DataFrame(x_)  
    df1['y'] = pd.DataFrame(y_)       
    freq_axis = np.fft.fftfreq(df.iloc[:,0].count())
    power = np.fft.fft(df['y']).real
    power_freq = pd.DataFrame(power[:], columns=['power'])
    print(len(power_freq))
#    plt.plot(freq_axis[:], power_freq)
#    plt.show()
#    plt.xlim(40,180 )
#    plt.ylim(-.1,1)
    pad = np.arange(25,180,1)
    power_freq_pad = pd.concat([power_freq, pd.DataFrame(index = pad)], axis = 0, sort=False)
    power_freq_pad.fillna(0,inplace = True)
    power_freq_scale = scaler.fit_transform(power_freq_pad)
    df2 = pd.DataFrame(power_freq_scale)
    plt.plot(power_freq_pad)
    fft_profile = df2[:180].transpose()
    fft_profile['device'] = 'ring_resonator_single_mode'
    fft_profile['asym'] = stats.skew(df.y)
    fft_profile['num_peaks'] = len(indexes)
    fft_profile['Q'] = center_wavelength/FWHM
    print(fft_profile)
#    fft_decomp.append(fft_profile)
    fft_profile.to_csv('fft'+fname)
    
    

    
#df_q = pd.DataFrame({'filnames':file_names,'fft':fft_p ,'quality_factor':Q, 'skew':asym, 'number_of_peaks': number_of_peaks})
#df_q.to_csv('peak_characteristics')
