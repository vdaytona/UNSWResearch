'''
Created on 26 Apr 2016
demonstration for matlab result

Fitting multiple gaussian

@author: Daytona
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import copy
from __builtin__ import str


def guassion_fun(xData, wavelength, intensity, FWHM):
    return intensity * np.exp((-1 * np.power((xData - wavelength),2) / np.power(FWHM, 2)))
          
def fitted_total_spectrum(xData, center_wavelength, center_intensity, FWHM):
    y = []
    for i in range(len(center_wavelength)) :
        y.append(guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i]))
    fitted = y[0]
    for i in range(1,len(y)) :
        fitted += y[i]
    return fitted
    
if __name__ == '__main__':
    # 1. import data
    rawData = pd.read_csv("./Data/input1.csv")
        
    xData = np.float64(rawData["Wavelength"][:701].values)
    yData = np.float64(rawData["Intensity"][:701].values)
    
    ######
    #plot#
    ######
    plt.figure(figsize=(12,10))
    plt.plot(xData, yData, label = "real")
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a. u.)")
    plt.show()
    
    # fitting parameter from Matlab
    center_wavelength = [944.1,1207,1402,1424,1542]
    FWHM = [42.22,143.1,92.99,25.39,21.49]
    center_intensity = [0.09989,0.225,0.7958,0.212,0.1197]
    
    fitted = fitted_total_spectrum(xData, center_wavelength, center_intensity, FWHM)
       
    
    ######
    #plot#
    ######
    plt.figure(figsize=(12,10))
    plt.plot(xData, yData, label = "real")
    plt.plot(xData, fitted, label = "fitted")
    for i in range(len(center_intensity)) :
        y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
        plt.plot(xData,y,label = str(i))
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a. u.)")
    plt.show()
    
    