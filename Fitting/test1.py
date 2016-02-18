'''
Created on 18 Feb 2016

@author: purewin7
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
#===============================================================================
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
# 
# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# ydata = y + 0.2 * np.random.normal(size=len(xdata))
# 
# popt, pcov = curve_fit(func, xdata, ydata)
# 
# #plt.plot(xdata,y)
# plt.plot(xdata,ydata)
# plt.show()
# 
# print popt
# print pcov
#===============================================================================
#===============================================================================
# def func(x, a,b,c,d):
#     return a+(b/(c*np.power(np.pi/2,0.5))) * np.exp(-2 * np.power(((x-d)/c),2))
#===============================================================================

def func(x, a,b,c):
    return a * np.exp(-1 * np.power(((x-b)/c),2))

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 /wid)


    
rawData = pd.read_csv("./input.csv")
startWavelength = 1400
stopWavelength = 1450
startIndex = np.where(rawData["Wavelength"]==startWavelength)[0]
stopIndex = np.where(rawData["Wavelength"]==stopWavelength)[0] + 1
xdata = rawData["Wavelength"][startIndex : stopIndex].values
ydata = rawData["Intensity"][startIndex : stopIndex].values

print gaussian(10,1,5,2)


print len(xdata)
print len(ydata)

y = gaussian(xdata, 0.9923, 1427, 76.51)

plt.plot(xdata,y)
plt.show()

parameter = curve_fit(gaussian, xdata, ydata)

print parameter






