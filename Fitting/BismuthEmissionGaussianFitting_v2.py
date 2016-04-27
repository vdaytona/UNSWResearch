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


def fitting_process(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, threshold):
    # 1. calculate the initial r2_score
    fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
    
    r2_score_series = []
    coeff_score_series = []
    
    #===========================================================================
    # plt.plot(wavelength, real_intensity, label = "real")
    # plt.plot(wavelength, fitted, label = "fitted")
    # plt.legend()
    #===========================================================================
    #plt.show()
    previous_universe_coeff_score = 0.0
    previous_universe_r2_score = 0.0
    for i in range(10) :
        # fitting use coeff
        #previous_universe_coeff_score = 
        
        #=======================================================================
        # coeff_score = coeff_cal(real_intensity, fitted)
        # previous = 0.0
        # while coeff_score != previous :
        #     previous = copy.deepcopy(coeff_score)
        #     result = \
        #     universe_optimization_coeff(wavelength, real_intensity, center_wavelength, center_intensity, FWHM)
        #     center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
        #     fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
        #     coeff_score = coeff_cal(real_intensity, fitted)
        # print "coeff : " + str(coeff_score)
        # coeff_score_series.append(coeff_score)
        #=======================================================================
        
        # fitting with r2_score
        
        r2_score = r_square_score(real_intensity, fitted)
        previous = 0.0
        while r2_score != previous :
            previous = copy.deepcopy(r2_score)
            result = \
            universe_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM)
            center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
            fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
            r2_score = r_square_score(real_intensity, fitted)
        print "r2_score : " + str(r2_score)
        r2_score_series.append(r2_score)
    #===========================================================================
    # previous = 0.0
    # start_wavelength = 1400
    # stop_wavelength = 1440
    # start_index = wavelength.tolist().index(start_wavelength)
    # stop_index = wavelength.tolist().index(stop_wavelength)
    # r2_score = r_square_score(real_intensity[start_index:stop_index], fitted[start_index:stop_index])
    # while r2_score != previous :
    #     previous = copy.deepcopy(r2_score)
    #     result = \
    #     part_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, start_wavelength, stop_wavelength)
    #     center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
    #     fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
    #     r2_score = r_square_score(real_intensity[start_index:stop_index], fitted[start_index:stop_index])
    #     print r2_score
    #===========================================================================
    
    #===========================================================================
    # previous = 0.0
    # start_wavelength = 1520
    # stop_wavelength = 1550
    # start_index = wavelength.tolist().index(start_wavelength)
    # stop_index = wavelength.tolist().index(stop_wavelength)
    # r2_score = r_square_score(real_intensity[start_index:stop_index], fitted[start_index:stop_index])
    # while r2_score != previous :
    #     previous = copy.deepcopy(r2_score)
    #     result = \
    #     part_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, start_wavelength, stop_wavelength)
    #     center_wavelength, center_intensity, FWHM = result[0],result[1], result[2]
    #     fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
    #     r2_score = r_square_score(real_intensity[start_index:stop_index], fitted[start_index:stop_index])
    #     print r2_score
    #===========================================================================
    

    fitted = fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)
    r2_score = r_square_score(real_intensity, fitted)
    for i in range(len(center_wavelength)) :
        print str(center_wavelength[i]) + " , " + str(center_intensity[i]) + " , " + str(FWHM[i])
    #plt.plot(wavelength, real_intensity, label = "real")
    #plt.plot(wavelength, fitted, label = "fitted")
    for i in range(len(center_intensity)) :
        y = guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i])
        #plt.plot(xData,y,label = str(i))
    #plt.legend()
    #plt.show()
    
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
        if center_wavelength[i] != 1420 :
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
            #print minus_FWHM
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
        if center_wavelength[i] != 1420 :
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

def part_optimization(wavelength, real_intensity, center_wavelength, center_intensity, FWHM, start_wavelength, stop_wavelength):
    start_index = wavelength.tolist().index(start_wavelength)
    stop_index = wavelength.tolist().index(stop_wavelength)
    for i in range(len(center_intensity)):
        initial_score = \
        r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)[start_index:stop_index])
        optimize_score = copy.deepcopy(initial_score)
        optimize_center_intensity = copy.deepcopy(center_intensity)
        step = 0.01
        
        plus_center_intensity = copy.deepcopy(center_intensity)
        plus_center_intensity[i] = plus_center_intensity[i] * (1 + step)
        plus_score = \
        r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, plus_center_intensity, FWHM)[start_index:stop_index])
        minus_center_intensity = copy.deepcopy(center_intensity)
        minus_center_intensity[i] = minus_center_intensity[i] * (1 - step)
        minus_score = \
        r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, minus_center_intensity, FWHM)[start_index:stop_index])
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
        
    #===========================================================================
    # for i in range(len(center_wavelength)):
    #     if center_wavelength[i] != 1420 :
    #         initial_score = \
    #         r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)[start_index:stop_index])
    #         optimize_score = copy.deepcopy(initial_score)
    #         optimize_center_wavelength = copy.deepcopy(center_wavelength)
    #         step = 1
    #         plus_center_wavelength = copy.deepcopy(center_wavelength)
    #         plus_center_wavelength[i] = plus_center_wavelength[i] + step
    #         plus_score = \
    #         r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, plus_center_wavelength, center_intensity, FWHM)[start_index:stop_index])
    #         minus_center_wavelength = copy.deepcopy(center_wavelength)
    #         minus_center_wavelength[i] = minus_center_wavelength[i] - step
    #         minus_score = \
    #         r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, minus_center_wavelength, center_intensity, FWHM)[start_index:stop_index])
    #         
    #         if plus_score > minus_score :
    #             optimize_score = plus_score
    #             optimize_center_wavelength = plus_center_wavelength
    #         else :
    #             optimize_center_wavelength = minus_center_wavelength
    #             optimize_score = minus_score
    #         if optimize_score > initial_score :
    #             center_wavelength = optimize_center_wavelength
    #===========================================================================
            
    for i in range(len(FWHM)) :
            initial_score = \
            r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, center_intensity, FWHM)[start_index:stop_index])
            optimize_score = copy.deepcopy(initial_score)
            optimize_FWHM = copy.deepcopy(FWHM)
            step = 0.1
            plus_FWHM = copy.deepcopy(FWHM)
            plus_FWHM[i] = plus_FWHM[i] + step
            plus_score = \
            r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, center_intensity, plus_FWHM)[start_index:stop_index])
            minus_FWHM = copy.deepcopy(FWHM)
            minus_FWHM[i] = minus_FWHM[i] - step
            minus_score = \
            r_square_score(real_intensity[start_index:stop_index], fitted_total_spectrum(wavelength, center_wavelength, center_intensity, minus_FWHM)[start_index:stop_index])
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
    # target1 = target[0:370]
    # fitted1 = fitted[0:370]
    # target2 = target[415:550]
    # fitted2 = fitted[415:550]
    # target3 = target[550:650]
    # fitted3 = fitted[550:650]
    # score1 = r2_score(target1, fitted1) * 175 / 620
    # score2 = r2_score(target2, fitted2) * 300 / 620
    # score3 = r2_score(target3, fitted3) * 145 / 620
    # return score1 + score2 + score3
    #===========================================================================
    
    target1 = target[0:370]
    fitted1 = fitted[0:370]
    target2 = target[415:450]
    fitted2 = fitted[415:450]
    target3 = target[450:520]
    fitted3 = fitted[450:520]
    target4 = target[520:550]
    fitted4 = fitted[520:550]
    target5 = target[550:650]
    fitted5 = fitted[550:650]
    score1 = r2_score(target1, fitted1) * 25 / 80
    score2 = r2_score(target2, fitted2) * 25 / 80
    score3 = r2_score(target3, fitted3) * 12 / 80
    score4 = r2_score(target4, fitted4) * 10 / 80
    score5 = r2_score(target5, fitted5) * 6 / 80
    return score1 + score2 + score3 + score4 + score5
    
    #return r2_score(target, fitted)

def coeff_cal(target,fitted):
    return np.corrcoef(target, fitted)[0,1]
    #===========================================================================
    # target1 = target[0:370]
    # fitted1 = fitted[0:370]
    # target2 = target[415:450]
    # fitted2 = fitted[415:450]
    # target3 = target[450:515]
    # fitted3 = fitted[450:515]
    # target4 = target[515:550]
    # fitted4 = fitted[515:550]
    # target5 = target[550:650]
    # fitted5 = fitted[550:650]
    # score1 = np.corrcoef(target1, fitted1)[0,1] * 25 / 80
    # score2 = np.corrcoef(target2, fitted2)[0,1] * 25 / 80
    # score3 = np.corrcoef(target3, fitted3)[0,1] * 12 / 80
    # score4 = np.corrcoef(target4, fitted4)[0,1] * 12 / 80
    # score5 = np.corrcoef(target5, fitted5)[0,1] * 6 / 80
    # return score1 + score2 + score3 + score4 + score5
    #===========================================================================

def fitted_total_spectrum(xData, center_wavelength, center_intensity, FWHM):
    y = []
    for i in range(len(center_wavelength)) :
        y.append(guassion_fun(xData, center_wavelength[i], center_intensity[i], FWHM[i]))
    fitted = y[0]
    for i in range(1,len(y)) :
        fitted += y[i]
    return fitted

def fit_one_spectrum(xData, yData):
    # 2. set initial parameter
    
    #===========================================================================
    # FWHM = [113,100,86,20,85]
    # center_wavelength = [1091,1250,1420,1545, 1615]
    #===========================================================================
    
    FWHM = [113,100,86,20]
    center_wavelength = [1150,1310,1420,1545]
    intensity = []
    
    for i in range(len(center_wavelength)) :
        intensity.append(yData[xData.tolist().index(center_wavelength[i])])
    
    for i in range(len(center_wavelength)) :
        print str(center_wavelength[i]) + " , " + str(intensity[i]) + " , " + str(FWHM[i])
        
    # 3. fitting
    
    return fitting_process(xData, yData, center_wavelength, intensity, FWHM, 0.9959)
    
if __name__ == '__main__':
    # 1. import data
    # 1. read csv
    rawData = pd.read_csv("./Data/input_time.csv")
    
    #print rawData
    result = []
    xData = np.float64(rawData["Wavelength"][:650].values)
    yData = np.float64(rawData["0"][:650].values)
    spectrum_0 = fit_one_spectrum(xData, yData)
    yData = np.float64(rawData["10"][:650].values)
    spectrum_10 = fit_one_spectrum(xData, yData)
    yData = np.float64(rawData["20"][:650].values)
    spectrum_20 = fit_one_spectrum(xData, yData)
    yData = np.float64(rawData["40"][:650].values)
    spectrum_40 = fit_one_spectrum(xData, yData)
    yData = np.float64(rawData["60"][:650].values)
    spectrum_60 = fit_one_spectrum(xData, yData)
    
    result = [spectrum_0,spectrum_10,spectrum_20,spectrum_40,spectrum_60]
    
    maxium_intensity_Si = []
    maxium_intensity_P = []
    maxium_intensity_Al = []
    for i in range(len(result)) :
        maxium_intensity_Al.append(result[i][0])
        maxium_intensity_P.append(result[i][1])
        maxium_intensity_Si.append(result[i][2])
    print result[i][0]
    fig1 = plt
    fig1.plot(range(len(result)), maxium_intensity_Al,'o', label = "Al")
    fig1.plot(range(len(result)), maxium_intensity_P,'o', label = "P")
    fig1.plot(range(len(result)), maxium_intensity_Si,'o', label = "Si")
    fig1.legend()
    fig1.show()
    
    subtracted = rawData.copy()
    finalResult = rawData.copy()
    
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outputFileName = ("./Result/Gaussian-Fitting-%s.csv" % currentTime)
    
    