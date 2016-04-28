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

def func_photobleaching(x, a,b,c,d):
    return a + b * (1-np.exp(-1 * np.power((x/c),d)))


    

xdata = [20,30,60,120,240,360,540,720,1140,2880]
ydata = [9.870411156,9.914331068,10.04497169,10.10510634,10.16162174,10.18265744,10.25691759,10.30021202,10.40285453,10.43363305]

parameter = curve_fit(func_photobleaching, xdata, ydata)

print parameter






