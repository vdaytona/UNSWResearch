'''
Created on 18 Feb 2016

@author: purewin7
'''
import pandas as pd
import datetime
import numpy as np
import GaussianFit as gf
import matplotlib.pyplot as plt

def select_XY(xData, yData, startWavelength, stopWavelength):
    startIndex = np.where(rawData["Wavelength"]==startWavelength)[0]
    stopIndex = np.where(rawData["Wavelength"]==stopWavelength)[0] + 1
    x = xData[startIndex : stopIndex].values
    y = yData[startIndex : stopIndex].values
    return x,y


if __name__ == '__main__':
    # 1. read csv
    rawData = pd.read_csv("./Data/input1.csv")
    print rawData
    
    xData = rawData["Wavelength"]
    yData = rawData["Intensity"]
    
    subtracted = rawData.copy()
    finalResult = rawData.copy()
    
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outputFileName = ("./Result/Gaussian-Fitting-%s.csv" % currentTime)
    
    
    
    
    # 2. Loop
    ifContinue = True
    finalResult["Combined"] = 0
       
    while ifContinue == True:
        plt.plot(subtracted["Wavelength"],subtracted["Intensity"])
        plt.show()
        # 2.1 input center
        cen = input("Select center wavelength :")
        
        # 2.2 input startWavelength and stopWavelength for fitting
        startWavelength = input("Enter startWavelength :")
        stopWavelength = input("Enter stopWavelength :")
        
        # 2.3 select xData, ydata
        
        rawX, rawY = select_XY(xData, subtracted["Intensity"], startWavelength, stopWavelength)
        
        # 2.4 Gaussian Fitting
        parameter = gf.Gaussian_Fit(rawX, rawY, cen,ifPlot=True)[0]
        
        fittedY = gf.gaussian(xData, parameter["amp"], parameter["cen"], parameter["wid"])
        
        # 2.5 change fit into DataFrame, wavelength as index
        finalResult["Intensity_%s" %cen] = fittedY
        finalResult["Combined"] += fittedY
        
        # 2.6 subtract fitted data
        subtracted["Intensity"] -= fittedY
        
        # 2.7 plot fitted result
        plt.plot(rawData["Wavelength"],rawData["Intensity"])
        plt.plot(rawData["Wavelength"],subtracted["Intensity"])
        plt.plot(rawData["Wavelength"], fittedY)
        plt.show()
        
        # 2.8 ask continue?
        ifContinue = input("Continue (Y/N) ?  : ")
    finalResult
    finalResult.to_csv(outputFileName)
    
    print ("finish")
    
    # 3. save data / picture into csv wavelength, raw, each peak, and combined fit
    
    
    
    