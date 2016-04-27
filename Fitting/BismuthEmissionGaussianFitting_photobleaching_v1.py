'''
Created on 26 Apr 2016

Fitting multiploe gaussian

@author: Daytona
'''

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import copy
from __builtin__ import str


def fitting_process(wavelength, real_intensity, center_wavelength, center_intensity, FWHM):
    # 1. calculate the initial r2_score
    fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
        
    r2_score = r_square_score(real_intensity, fitted)
    previous = 0.0
    
    # iteration until get same r2 score
    while r2_score != previous :
        previous = copy.deepcopy(r2_score)
        result = \
        universe_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM)
        center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
        fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
        r2_score = r_square_score(real_intensity, fitted)
    print "r2_score : " + str(r2_score)
    
    
    ######
    #plot#
    ######
    
    #===========================================================================
    # plt.plot(wavelength, real_intensity, label = "real")
    # plt.plot(wavelength, fitted, label = "fitted")
    # for i in range(len(center_intensity)) :
    #     y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
    #     plt.plot(xData,y,label = str(i))
    # plt.legend()
    # plt.show()
    #===========================================================================
    
    return center_intensity

def universe_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM):
    for i in range(len(center_intensity)):
        initial_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
        optimize_score = copy.deepcopy(initial_score)
        optimize_center_intensity = copy.deepcopy(center_intensity)
        step = 0.01
        
        plus_center_intensity = copy.deepcopy(center_intensity)
        plus_center_intensity[i] = plus_center_intensity[i] * (1 + step)
        plus_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, plus_center_intensity, FWHM))
        minus_center_intensity = copy.deepcopy(center_intensity)
        minus_center_intensity[i] = minus_center_intensity[i] * (1 - step)
        minus_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, minus_center_intensity, FWHM))
        if plus_score > minus_score :
            optimize_score = plus_score
            optimize_center_intensity = plus_center_intensity
        else :
            optimize_center_intensity = minus_center_intensity
            optimize_score = minus_score
        if optimize_score > initial_score :
            center_intensity = optimize_center_intensity
        
    for i in range(len(center_wavelength)):
        if i !=8 :
            initial_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
            optimize_score = copy.deepcopy(initial_score)
            optimize_center_wavelength = copy.deepcopy(center_wavelength)
            step = 1
             
            plus_center_wavelength = copy.deepcopy(center_wavelength)
            plus_center_wavelength[i] = plus_center_wavelength[i] + step
            plus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, plus_center_wavelength, center_intensity, FWHM))
            minus_center_wavelength = copy.deepcopy(center_wavelength)
            minus_center_wavelength[i] = minus_center_wavelength[i] - step
            minus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, minus_center_wavelength, center_intensity, FWHM))
             
            if plus_score > minus_score :
                optimize_score = plus_score
                optimize_center_wavelength = plus_center_wavelength
            else :
                optimize_center_wavelength = minus_center_wavelength
                optimize_score = minus_score
            if optimize_score > initial_score :
                center_wavelength = optimize_center_wavelength
            
    for i in range(len(FWHM)) :
            initial_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
            optimize_score = copy.deepcopy(initial_score)
            optimize_FWHM = copy.deepcopy(FWHM)
            step = 1
            
            plus_FWHM = copy.deepcopy(FWHM)
            plus_FWHM[i] = plus_FWHM[i] + step
            plus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, plus_FWHM))
            minus_FWHM = copy.deepcopy(FWHM)
            minus_FWHM[i] = minus_FWHM[i] - step
            minus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, minus_FWHM))
            if plus_score > minus_score :
                optimize_score = plus_score
                optimize_FWHM = plus_FWHM
            else :
                optimize_FWHM = minus_FWHM
                optimize_score = minus_score
            if optimize_score > initial_score :
                FWHM = optimize_FWHM
    return [center_wavelength, center_intensity, FWHM]



def guassion_fun(xData, wavelength, intensity, FWHM):
    return intensity * np.exp((-1 * np.power((xData - wavelength),2) / np.power(FWHM, 2)))

def r_square_score(target,fitted):    
    #===========================================================================
    # for no loading, wavelength 1 nm interval 
    # target1 = target[0:370]
    # fitted1 = fitted[0:370]
    # target2 = target[415:450]
    # fitted2 = fitted[415:450]
    # target3 = target[450:520]
    # fitted3 = fitted[450:520]
    # target4 = target[520:550]
    # fitted4 = fitted[520:550]
    # target5 = target[550:650]
    # fitted5 = fitted[550:650]
    #===========================================================================
    
    # for h2 loaded, wavelength 1.1 nm interval 
    target1 = target[0:int(370/1.1)]
    fitted1 = fitted[0:int(370/1.1)]
    target2 = target[int(415/1.1):int(450/1.1)]
    fitted2 = fitted[int(415/1.1):int(450/1.1)]
    target3 = target[int(450/1.1):int(520/1.1)]
    fitted3 = fitted[int(450/1.1):int(520/1.1)]
    target4 = target[int(520/1.1):int(550/1.1)]
    fitted4 = fitted[int(520/1.1):int(550/1.1)]
    target5 = target[int(550/1.1):int(650/1.1)]
    fitted5 = fitted[int(550/1.1):int(650/1.1)]
    
    
    score1 = r2_score(target1, fitted1) * 35 / 80
    score2 = r2_score(target2, fitted2) * 25 / 80
    score3 = r2_score(target3, fitted3) * 14 / 80
    score4 = r2_score(target4, fitted4) * 10 / 80
    score5 = r2_score(target5, fitted5) * 6 / 80
    return r2_score(target, fitted)
    return score1 + score2 + score3 + score4 + score5


def fitted_total_spectrum(xData, center_wavelength, center_intensity, FWHM):
    y = []
    for i in range(len(center_wavelength)) :
        y.append(guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i]))
    fitted = y[0]
    for i in range(1,len(y)) :
        fitted += y[i]
    return fitted

def fit_one_spectrum(xData, yData):
    # set initial parameter
    
    #===========================================================================
    # FWHM = [113,100,86,20,85]
    # center_wavelength = [1150,1310,1420,1545, 1615]
    #===========================================================================
    
    FWHM = [113,100,86,20]
    center_wavelength = [1091,1250,1426,1535]
    
    intensity = []    
    for i in range(len(center_wavelength)) :
        index = ( center_wavelength[i] - 1000 ) / 1.1
        intensity.append(yData[index])
        
    # 3. fitting
    
    result = fitting_process(xData, yData, center_wavelength, intensity, FWHM)
    print result
    
    return result
    
if __name__ == '__main__':
    # 1. import data
    #rawData = pd.read_csv("./Data/input_time.csv")
    
    rawData = pd.read_csv("./Data/input_time_h2loaded.csv")
    
    #time = range(1, 20)
    #time.extend(range(20,65,5))
    
    time = range(31)
    
    xData = np.float64(rawData["Wavelength"][:593].values)
    result = []
    
    maxIntensity = max(rawData["0"])
    
    for t in range(len(time)):
        print "time : " + str(time[t])
        #yData = np.float64(rawData[str(time[t])][:650].values)
        yData = np.float64(rawData[str(time[t])][:593].values / maxIntensity)
        result.append(fit_one_spectrum(xData, yData))
    
    maxium_intensity_Si = []
    maxium_intensity_P = []
    maxium_intensity_Al = []
    maxium_intensity_Er = []
    
    for i in range(len(result)) :
        maxium_intensity_Al.append(result[i][0])
        maxium_intensity_P.append(result[i][1])
        maxium_intensity_Si.append(result[i][2])
        maxium_intensity_Er.append(result[i][3])
        
    normalized_maxium_intensity_Al = [float(i)/max(maxium_intensity_Al) for i in maxium_intensity_Al]
    normalized_maxium_intensity_P = [float(i)/max(maxium_intensity_P) for i in maxium_intensity_P]
    normalized_maxium_intensity_Si = [float(i)/max(maxium_intensity_Si) for i in maxium_intensity_Si]
    normalized_maxium_intensity_Er = [float(i)/max(maxium_intensity_Er) for i in maxium_intensity_Er]
    fig1 = plt
    fig1.plot(time, normalized_maxium_intensity_Al,'o', label = "Al")
    fig1.plot(time, normalized_maxium_intensity_P,'o', label = "P")
    fig1.plot(time, normalized_maxium_intensity_Si,'o', label = "Si")
    fig1.plot(time, normalized_maxium_intensity_Er,'o', label = "Er")
    fig1.ylim(0.3,1.1)
    fig1.xlim(-1,35)
    fig1.legend()
    fig1.show()
    
    