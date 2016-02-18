'''
Created on 18 Feb 2016

@author: purewin7
'''
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# x = data[1:, 0]
# y = data[1:, 1]
#===============================================================================

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))

def Gaussian_Fit(xData,yData, cen, ifPlot=False):

    gmod = Model(gaussian)
    result = gmod.fit(yData, x=xData, amp=5, cen=cen, wid=1)
    
    if ifPlot == True : 
        plt.plot(xData, yData,         'bo')
        plt.plot(xData, result.best_fit, 'r-')
        plt.ylim(min(min(result.best_fit),min(xData)),max(max(result.best_fit),max(yData)))
        plt.show()
    
    return result.best_values, result.best_fit