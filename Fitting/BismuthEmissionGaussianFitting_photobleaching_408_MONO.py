'''
Created on 26 Apr 2016

Fitting multiple gaussian

@author: Daytona
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sklearn.metrics as sm
import copy
from __builtin__ import str


def fitting_process(wavelength, real_intensity, center_wavelength, center_intensity, FWHM):
    
    fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
    
    coeff_score = coeff_cal(real_intensity, fitted)
    previous = 0.0
    while coeff_score != previous :
        previous = copy.deepcopy(coeff_score)
        result = \
        universe_optimization_coeff(wavelength, real_intensity, center_wavelength, center_intensity, FWHM)
        center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
        fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
        coeff_score = coeff_cal(real_intensity, fitted)
    print "coeff : " + str(coeff_score)
      
    ######
    #plot#
    ######
    plt.figure(figsize=(12,10))
    plt.plot(wavelength, real_intensity, label = "real")
    plt.plot(wavelength, fitted, label = "fitted")
    for i in range(len(center_intensity)) :
        y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
        plt.plot(xData,y,label = str(i))
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a. u.)")
    print sm.r2_score(real_intensity,fitted)
    plt.show()
    
    # 1. calculate the initial r2_score
    
        
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
    plt.figure(figsize=(12,10))
    plt.plot(wavelength, real_intensity, label = "real")
    plt.plot(wavelength, fitted, label = "fitted")
    for i in range(len(center_intensity)) :
        y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
        plt.plot(xData,y,label = str(i))
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a. u.)")
    print sm.r2_score(real_intensity,fitted)
    plt.show()
    
    print center_wavelength
    print center_intensity
    print FWHM
    
    
    
    for i in range(2) :
        coeff_score = coeff_cal(real_intensity, fitted)
        previous = 0.0
        while coeff_score != previous :
            previous = copy.deepcopy(coeff_score)
            result = \
            universe_optimization_coeff(wavelength, real_intensity, center_wavelength, center_intensity, FWHM)
            center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
            fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
            coeff_score = coeff_cal(real_intensity, fitted)
        print "coeff : " + str(coeff_score)
        
        ######
        #plot#
        ######
        
        plt.plot(wavelength, real_intensity, label = "real")
        plt.plot(wavelength, fitted, label = "fitted")
        for i in range(len(center_intensity)) :
            y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
            plt.plot(xData,y,label = str(i))
        plt.legend()
        plt.show()
        
        # 1. calculate the initial r2_score
        
            
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
        
        plt.plot(wavelength, real_intensity, label = "real")
        plt.plot(wavelength, fitted, label = "fitted")
        for i in range(len(center_intensity)) :
            y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
            plt.plot(xData,y,label = str(i))
        plt.legend()
        plt.show()
    
    
    
    return center_intensity

def universe_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM):
    for i in range(len(center_intensity)):
        initial_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
        optimize_score = copy.deepcopy(initial_score)
        optimize_center_intensity = copy.deepcopy(center_intensity)
        step = 0.05
        
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
        if i != 8 :
            initial_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
            optimize_score = copy.deepcopy(initial_score)
            optimize_center_wavelength = copy.deepcopy(center_wavelength)
            step = 0.01
             
            plus_center_wavelength = copy.deepcopy(center_wavelength)
            plus_center_wavelength[i] = plus_center_wavelength[i] * (1 + step)
            plus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, plus_center_wavelength, center_intensity, FWHM))
            minus_center_wavelength = copy.deepcopy(center_wavelength)
            minus_center_wavelength[i] = minus_center_wavelength[i]* (1 - step)
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
            step = 0.01
            
            plus_FWHM = copy.deepcopy(FWHM)
            plus_FWHM[i] = plus_FWHM[i] *  (1 + step)
            plus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, plus_FWHM))
            minus_FWHM = copy.deepcopy(FWHM)
            minus_FWHM[i] = minus_FWHM[i] * (1 - step)
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

def universe_optimization_coeff(wavelength, real_intensity, center_wavelength, center_intensity, FWHM):
    for i in range(len(center_intensity)):
        initial_score = \
        coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
        optimize_score = copy.deepcopy(initial_score)
        optimize_center_intensity = copy.deepcopy(center_intensity)
        step = 0.01
        
        plus_center_intensity = copy.deepcopy(center_intensity)
        plus_center_intensity[i] = plus_center_intensity[i] * (1 + step)
        plus_score = \
        coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, plus_center_intensity, FWHM))
        minus_center_intensity = copy.deepcopy(center_intensity)
        minus_center_intensity[i] = minus_center_intensity[i] * (1 - step)
        minus_score = \
        coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, minus_center_intensity, FWHM))
        #=======================================================================
        # print plus_score, minus_score , initial_score
        # print center_intensity
        #=======================================================================
        if plus_score > minus_score :
            optimize_score = plus_score
            optimize_center_intensity = plus_center_intensity
        else :
            optimize_center_intensity = minus_center_intensity
            optimize_score = minus_score
        if optimize_score > initial_score :
            center_intensity = optimize_center_intensity
            
        #=======================================================================
        # print plus_score, minus_score , initial_score
        # print center_intensity
        #=======================================================================
        
    for i in range(len(center_wavelength)):
        if i != 8 :
            initial_score = \
            coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
            optimize_score = copy.deepcopy(initial_score)
            optimize_center_wavelength = copy.deepcopy(center_wavelength)
            step = 1
            plus_center_wavelength = copy.deepcopy(center_wavelength)
            plus_center_wavelength[i] = plus_center_wavelength[i] + step
            plus_score = \
            coeff_cal(real_intensity, fitted_total_spectrum(wavelength, plus_center_wavelength, center_intensity, FWHM))
            minus_center_wavelength = copy.deepcopy(center_wavelength)
            minus_center_wavelength[i] = minus_center_wavelength[i] - step
            minus_score = \
            coeff_cal(real_intensity, fitted_total_spectrum(wavelength, minus_center_wavelength, center_intensity, FWHM))
            
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
            coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM))
            optimize_score = copy.deepcopy(initial_score)
            optimize_FWHM = copy.deepcopy(FWHM)
            step = 1
            plus_FWHM = copy.deepcopy(FWHM)
            plus_FWHM[i] = plus_FWHM[i] + step
            plus_score = \
            coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, plus_FWHM))
            minus_FWHM = copy.deepcopy(FWHM)
            minus_FWHM[i] = minus_FWHM[i] - step
            #print minus_FWHM
            minus_score = \
            coeff_cal(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, minus_FWHM))
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
    # check no nan
    target = check_nan(target)
    fitted = check_nan(fitted)  
    return  r2_score(target[:425], fitted[:425])
    #for no loading, wavelength 1 nm interval
    
    target1 = target[350:470]
    fitted1 = fitted[350:470]
    target2 = target[515:550]
    fitted2 = fitted[515:550]
    target3 = target[550:620]
    fitted3 = fitted[550:620]
    target4 = target[620:666]
    fitted4 = fitted[620:666]
    target5 = target[650:700]
    fitted5 = fitted[650:700]
    target6 = target[0:350]
    fitted6 = fitted[0:350]
    score1 = r2_score(target1, fitted1) * 25 / 105
    score2 = r2_score(target2, fitted2) * 25 / 105
    score3 = r2_score(target3, fitted3) * 10 / 105
    score4 = r2_score(target4, fitted4) * 15 / 105
    score5 = r2_score(target5, fitted5) * 0 / 105
    score6 = r2_score(target6, fitted6) * 15 / 105
    
    #===========================================================================
    # print target5
    # print fitted5
    #===========================================================================
    
    # for h2 loaded, wavelength 1.1 nm interval 
    
    #===========================================================================
    # target1 = target[0:int(370/1.1)]
    # fitted1 = fitted[0:int(370/1.1)]
    # target2 = target[int(415/1.1):int(450/1.1)]
    # fitted2 = fitted[int(415/1.1):int(450/1.1)]
    # target3 = target[int(450/1.1):int(520/1.1)]
    # fitted3 = fitted[int(450/1.1):int(520/1.1)]
    # target4 = target[int(520/1.1):int(550/1.1)]
    # fitted4 = fitted[int(520/1.1):int(550/1.1)]
    # target5 = target[int(550/1.1):int(650/1.1)]
    # fitted5 = fitted[int(550/1.1):int(650/1.1)]
    #===========================================================================
        
    
    
    #===========================================================================
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
    # score1 = r2_score(target1, fitted1) * 25 / 90
    # score2 = r2_score(target2, fitted2) * 25 / 90
    # score3 = r2_score(target3, fitted3) * 12 / 90
    # score4 = r2_score(target4, fitted4) * 10 / 90
    # score5 = r2_score(target5, fitted5) * 6 / 90
    #===========================================================================
    
    return score1 + score2 + score3 + score4 + score5 + score6

def coeff_cal(target,fitted):
    return np.corrcoef(target, fitted)[0,1]
    #===========================================================================
    # target1 = target[0:470]
    # fitted1 = fitted[0:470]
    # target2 = target[515:]
    # fitted2 = fitted[515:]
    # score1 = np.corrcoef(target1, fitted1)[0,1]
    # score2 = np.corrcoef(target2, fitted2)[0,1]
    # 
    # return score1 + score2
    #===========================================================================
    

def check_nan(inputArray):
    for i in range(len(inputArray)):
        if np.isnan(inputArray[i]) :
            if i == 0 :
                np.put(inputArray, i, (inputArray[i+1]))
            else :
                np.put(inputArray, i, (inputArray[i-1] + inputArray[i+1])/2)
    return inputArray
            

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
    
    #===========================================================================
    # FWHM = [113,100,86,20]
    # center_wavelength = [1150,1310,1420,1545]
    #===========================================================================
    
    FWHM = [20,20,113,100,86,20]
    center_wavelength = [832,950,1090,1250,1420,1536]
    
    #===========================================================================
    # FWHM = [100,86,20]
    # center_wavelength = [1150,1426,1535]
    #===========================================================================
    
    intensity = []    
    for i in range(len(center_wavelength)) :
        intensity.append(yData[xData.tolist().index(center_wavelength[i])])
        
    # 3. fitting
    
    result = fitting_process(xData, yData, center_wavelength, intensity, FWHM)
    print result
    
    return result
    
if __name__ == '__main__':
    # 1. import data
    rawData = pd.read_csv("./Data/408excitation_1.csv")
        
    xData = np.float64(rawData["Wavelength"][:451].values)
    yData = np.float64(rawData["Intensity"][:451].values)
    
    print len(xData)
    fit_one_spectrum(xData, yData)
    
    