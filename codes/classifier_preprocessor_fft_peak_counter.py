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
path_dir = r'C:\Interpolation_Project\classification\raw_data\ring_resonator_classification\all_pass'


#loop over the files and then create a list of file names to later iterate over
    

filenames = []
Q = []
asym = []
number_of_peaks = []
fbg_decomp = []

# cols = ['x', 'y']
for  fname in os.listdir(path_dir):
    print(fname)
    file_path = (os.path.join(path_dir, fname))
    df = pd.read_csv(file_path, sep = '\t', header = 8,  engine = 'python',usecols=[0,1])#, names =cols )
    df['x'], df['y']= df.iloc[:,0] , df.iloc[:,1]
    df.sort_values(by='x', ascending =True, inplace = True) 
    df.drop_duplicates( inplace =True)
    # df.plot('x','y') 
#    df['x_scale'] = minmax_scale(df.y, feature_range=(0,1))
#    df['y_scale'] = minmax_scale(df.x, feature_range=(0,1))    
#    indexes = peakutils.indexes(df.y, thres=0.1, min_dist=100)
#    pplot(df.x_scale, df.y_scale, indexes)
#    plt.show()
#    print(indexes)
    tck = interpolate.splrep(df.x,df.y,s=0.00000001) # s =m-sqrt(2m) where m= #datapts and s is smoothness factor
    x_ = np.arange (df.x.min(),df.x.max(), 0.003)
    y_ = interpolate.splev(x_, tck, der=0)
#    plt.plot(x_,y_)
    # plt.plot(df.x, df.y)
    HM =(np.max(y_)-np.min(y_))/2
    w = splrep(x_, y_ - HM, k=3)
#        print(sproot(w_j))
    try:
        if len(sproot(w))%2 == 0:
            r1 , r2 = sproot(w)
            # print(r1, r2)
            FWHM = np.abs(r1 - r2)
            center_wavelength = r1 + FWHM/2
            Q.append(center_wavelength/FWHM)
            skewness = stats.skew(y_)
            asym.append(skewness)    
#            file_names.append(fname)
            fnam = fname
            indexes = peakutils.indexes(y_, thres=0.5, min_dist=100)
#            pplot(x_,y_,indexes)
#            plt.show()
#            plt.xlim(1549,1553)
            number_of_peaks.append(len(indexes))
#            plt.plot(df['x'],df['y'])
#            plt.scatter(x_,y_)
#            plt.show()
            
    except (TypeError, ValueError):
        print(fname,'error')
        continue
    #output normalized fft profile
    freq_axis = np.fft.fftfreq(len(x_))
    power = np.fft.fft(y_).real
    df1 = pd.DataFrame(power)
    power_scale = scaler.fit_transform(df1)
    print(len(power_scale))
    # df2 = pd.DataFrame( { 'fname':{fname}, 'device':['fbg'], 'asym': [skewness], 'num_peaks':[len(indexes)], 'Q' : [center_wavelength/FWHM]})
    power_freq = pd.DataFrame(power_scale, columns=['power'])
#    plt.plot(power_freq)
    fft_profile = power_freq[:180].transpose()
    fft_profile.reset_index(drop=True, inplace=True)
    # df3 = pd.concat([ fft_profile , df2], join_axes = 'fname')
    plt.plot(freq_axis[:], power_scale[:])
    fft_profile['device'] = 'fbg'
    fft_profile['asym'] = skewness
    fft_profile['num_peaks'] = 1
    fft_profile['Q'] = center_wavelength/FWHM 
#    plt.xlim(0,.05 )
#    plt.ylim(-.1,100)
    print(fft_profile)  
    # fft_profile.to_csv('fft'+fname)
    
    

