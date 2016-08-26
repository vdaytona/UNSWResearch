'''
Created on 26 Apr 2016

Fitting multiple gaussian
using weight array to adjust the weight:
to ensure each spectrum has same weight

@author: Daytona
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sklearn.metrics as sm
import copy
from __builtin__ import str


def fitting_process(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, weight_list):
    
    fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
    weight_list = updata_weight_list(wavelength, center_wavelength, center_intensity, FWHM, weight_list)
    print weight_list
    
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
    #===========================================================================
    # plt.figure(figsize=(12,10))
    # plt.plot(wavelength, real_intensity, label = "real")
    # plt.plot(wavelength, fitted, label = "fitted")
    # for i in range(len(center_intensity)) :
    #     y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
    #     plt.plot(xData,y,label = str(i))
    # plt.legend()
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Intensity (a. u.)")
    # print sm.r2_score(real_intensity,fitted)
    # plt.show()
    #===========================================================================
    
    # 1. calculate the initial r2_score
    
        
    r2_score = r_square_score(real_intensity, fitted, weight_list)
    previous = 0.0
    weight_list = updata_weight_list(wavelength, center_wavelength, center_intensity, FWHM, weight_list)
    # iteration until get same r2 score
    while r2_score != previous :
        previous = copy.deepcopy(r2_score)
        result = \
        universe_optimization_new(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, weight_list)
        center_wavelength, center_intensity, FWHM, weight_list = result[0],result[1], result[2], result[3]
        weight_list = updata_weight_list(wavelength, center_wavelength, center_intensity, FWHM, weight_list)
        fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
        r2_score = r_square_score(real_intensity, fitted, weight_list)
    print "r2_score : " + str(r2_score)
    
    
    ######
    #plot#
    ######
    #===========================================================================
    # plt.figure(figsize=(12,10))
    # plt.plot(wavelength, real_intensity, label = "real")
    # plt.plot(wavelength, fitted, label = "fitted")
    # for i in range(len(center_intensity)) :
    #     y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
    #     plt.plot(xData,y,label = str(i))
    # plt.legend()
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Intensity (a. u.)")
    # print sm.r2_score(real_intensity,fitted)
    # plt.show()
    #===========================================================================
    
    print center_wavelength
    print center_intensity
    print FWHM
    
    for i in range(2) :
        print i
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
        
        #=======================================================================
        # plt.plot(wavelength, real_intensity, label = "real")
        # plt.plot(wavelength, fitted, label = "fitted")
        # for i in range(len(center_intensity)) :
        #     y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
        #     plt.plot(xData,y,label = str(i))
        # plt.legend()
        # plt.show()
        #=======================================================================
        
        # 1. calculate the initial r2_score
        
        weight_list = updata_weight_list(wavelength, center_wavelength, center_intensity, FWHM, weight_list)
        r2_score = r_square_score(real_intensity, fitted,weight_list)
        previous = 0.0
        
        # iteration until get same r2 score
        while r2_score != previous :
            previous = copy.deepcopy(r2_score)
            result = \
            universe_optimization_new(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, weight_list)
            center_wavelength, center_intensity, FWHM, weight_list = result[0],result[1], result[2], result[3]
            weight_list = updata_weight_list(wavelength, center_wavelength, center_intensity, FWHM, weight_list)
            fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
            r2_score = r_square_score(real_intensity, fitted,weight_list)
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

def universe_optimization_newnew(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, weight_list):
    initial_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM),weight_list)
    original_parameter = [center_wavelength, center_intensity, FWHM]
    parameter_set = []
    score_set = []
    step = 0.01
    
    for i in range(3):
        for j in range(len(center_intensity)) :
            new_parameter = copy.deepcopy(original_parameter)
            new_parameter[i][j] = new_parameter[i][j] * (1 + step)
            new_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, new_parameter[0], new_parameter[1], new_parameter[2]),weight_list)
            score_set.append(new_score)
            parameter_set.append(new_parameter)
            new_parameter = copy.deepcopy(original_parameter)
            new_parameter[i][j] = new_parameter[i][j] * (1 - step)
            new_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, new_parameter[0], new_parameter[1], new_parameter[2]),weight_list)
            score_set.append(new_score)
            parameter_set.append(new_parameter)
    max_new_score = max(score_set)
    #print score_set
    if initial_score > max_new_score :
        return [center_wavelength, center_intensity, FWHM, weight_list]
    else:
        index_best = score_set.index(max_new_score)
        best_parameter = parameter_set[index_best]
        best_parameter.append(weight_list)
        return best_parameter

def universe_optimization_new(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, weight_list):
    initial_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM),weight_list)
    original_parameter = [center_wavelength, center_intensity, FWHM]
    parameter_set = []
    score_set = []
    step = 0.01
    for i in range(3):
        for j in range(len(center_intensity)) :
            new_parameter = copy.deepcopy(original_parameter)
            new_parameter[i][j] = new_parameter[i][j] * (1 + step)
            new_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, new_parameter[0], new_parameter[1], new_parameter[2]),weight_list)
            score_set.append(new_score)
            parameter_set.append(new_parameter)
            new_parameter = copy.deepcopy(original_parameter)
            new_parameter[i][j] = new_parameter[i][j] * (1 - step)
            new_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, new_parameter[0], new_parameter[1], new_parameter[2]),weight_list)
            score_set.append(new_score)
            parameter_set.append(new_parameter)
    max_new_score = max(score_set)
    #print score_set
    if initial_score > max_new_score :
        return [center_wavelength, center_intensity, FWHM, weight_list]
    else:
        index_best = score_set.index(max_new_score)
        best_parameter = parameter_set[index_best]
        best_parameter.append(weight_list)
        return best_parameter

def universe_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, weight_list):
    for i in range(len(center_intensity)):
        initial_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM),weight_list)
        optimize_score = copy.deepcopy(initial_score)
        optimize_center_intensity = copy.deepcopy(center_intensity)
        step = 0.05
        
        plus_center_intensity = copy.deepcopy(center_intensity)
        plus_center_intensity[i] = plus_center_intensity[i] * (1 + step)
        plus_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, plus_center_intensity, FWHM),weight_list)
        minus_center_intensity = copy.deepcopy(center_intensity)
        minus_center_intensity[i] = minus_center_intensity[i] * (1 - step)
        minus_score = \
        r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, minus_center_intensity, FWHM),weight_list)
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
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM),weight_list)
            optimize_score = copy.deepcopy(initial_score)
            optimize_center_wavelength = copy.deepcopy(center_wavelength)
            step = 0.01
             
            plus_center_wavelength = copy.deepcopy(center_wavelength)
            plus_center_wavelength[i] = plus_center_wavelength[i] * (1 + step)
            plus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, plus_center_wavelength, center_intensity, FWHM),weight_list)
            minus_center_wavelength = copy.deepcopy(center_wavelength)
            minus_center_wavelength[i] = minus_center_wavelength[i]* (1 - step)
            minus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, minus_center_wavelength, center_intensity, FWHM),weight_list)
             
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
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM),weight_list)
            optimize_score = copy.deepcopy(initial_score)
            optimize_FWHM = copy.deepcopy(FWHM)
            step = 0.01
            
            plus_FWHM = copy.deepcopy(FWHM)
            plus_FWHM[i] = plus_FWHM[i] *  (1 + step)
            plus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, plus_FWHM),weight_list)
            minus_FWHM = copy.deepcopy(FWHM)
            minus_FWHM[i] = minus_FWHM[i] * (1 - step)
            minus_score = \
            r_square_score(real_intensity, fitted_total_spectrum(wavelength, center_wavelength, center_intensity, minus_FWHM),weight_list)
            if plus_score > minus_score :
                optimize_score = plus_score
                optimize_FWHM = plus_FWHM
            else :
                optimize_FWHM = minus_FWHM
                optimize_score = minus_score
            if optimize_score > initial_score :
                FWHM = optimize_FWHM
    
    return [center_wavelength, center_intensity, FWHM, weight_list]

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
    # using Lorentiz fitting
    #return intensity / (1 + np.power((xData - wavelength)/ (FWHM/2),2))

def r_square_score(target,fitted,weight_list):
    # check no nan
    target = check_nan(target)
    fitted = check_nan(fitted)
    #return r2_score(target[:-30], fitted[:-30], weight_list[:-30])
    
    target1 = target[0:465]
    fitted1 = fitted[0:465]
    weight1 = weight_list[0:465]
    target2 = target[495:-30]
    fitted2 = fitted[495:-30]
    weight2 = weight_list[495:-30]
    
    score1 = r2_score(target1, fitted1, weight1) * (460.0 / (460.0 + 720.0 - 495.0))
    score2 = r2_score(target2, fitted2, weight2) * ((720.0 - 495.0) / (460.0 + 720.0 - 495.0))
    return score1 + score2
    #for no loading, wavelength 1 nm interval
    
    #===========================================================================
    # target1 = target[350:450]
    # fitted1 = fitted[350:450]
    # target2 = target[486:550]
    # fitted2 = fitted[486:550]
    # target3 = target[550:620]
    # fitted3 = fitted[550:620]
    # target4 = target[620:650]
    # fitted4 = fitted[620:650]
    # target5 = target[650:700]
    # fitted5 = fitted[650:700]
    # target6 = target[0:350]
    # fitted6 = fitted[0:350]
    # weight1 = 35
    # weight2 = 35
    # weight3 = 25
    # weight4 = 25
    # weight5 = 0
    # weight6 = 25
    # weight = weight1 + weight2 + weight3 + weight4 + weight5 + weight6
    # 
    # score1 = r2_score(target1, fitted1) * weight1 / weight
    # score2 = r2_score(target2, fitted2) * weight2 / weight
    # score3 = r2_score(target3, fitted3) * weight3 / weight
    # score4 = r2_score(target4, fitted4) * weight4 / weight
    # score5 = r2_score(target5, fitted5) * weight5 / weight
    # score6 = r2_score(target6, fitted6) * weight6 / weight
    #===========================================================================
    
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
    
    #return score1 + score2 + score3 + score4 + score5 + score6

def coeff_cal(target,fitted):
    return np.corrcoef(target[:-30], fitted[:-30])[0,1]

    #===========================================================================
    # target1 = target[0:440]
    # fitted1 = fitted[0:440]
    # target2 = target[485:750]
    # fitted2 = fitted[485:750]
    # score1 = np.corrcoef(target1, fitted1)[0,1]
    # score2 = np.corrcoef(target2, fitted2)[0,1]
    #  
    # return (score1 + score2) / 2
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

def updata_weight_list(xData, center_wavelength, center_intensity, FWHM, weight_list):
    # get new weight list
    for i in range(len(xData)) :
        weight_list[i] = 0.0
    for i in range(len(center_wavelength)) :
        # e.x. the wavelength is from 1200 - 1250, the weight for each point will be  1 / (indexof(1500) - indexof(1200))
        new_spectrum = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
        spectrum_index_list = []
        for j in range(len(new_spectrum)) :
            if new_spectrum[j] > 0.01:
                spectrum_index_list.append(j)
        weight = 1.0 / len(spectrum_index_list)
        for j in range(len(weight_list)) :
            if j in spectrum_index_list :
                weight_list[j] += weight
    return weight_list

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
    
    FWHM = [20,113,100,86,20]
    center_wavelength = [950,1091,1250,1419,1535]
    
    #===========================================================================
    # FWHM = [100,86,20]
    # center_wavelength = [1150,1426,1535]
    #===========================================================================
    
    intensity = []    
    for i in range(len(center_wavelength)) :
        #intensity.append(0.1)
        intensity.append(yData[xData.tolist().index(center_wavelength[i])])
    weight_list = []
    for i in range(len(xData)) :
        weight_list.append(0)
    #intensity[3] = 0.02
    # 3. fitting
    
    result = fitting_process(xData, yData, center_wavelength, intensity, FWHM, weight_list)
    print result
    
    return result
    
if __name__ == '__main__':
    # 1. import data
    rawData = pd.read_csv("./Data/input1.csv")
        
    xData = np.float64(rawData["Wavelength"][:701].values)
    yData = np.float64(rawData["Intensity"][:701].values)
    
    print len(xData)
    fit_one_spectrum(xData, yData)
    
    