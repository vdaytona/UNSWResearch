'''
Created on 18 Feb 2016

@author: purewin7
'''
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rawData = pd.read_csv("./Data/input.csv")
x = rawData["Wavelength"].values
y = rawData["Intensity"].values

#===============================================================================
# startWavelength = 1400
# stopWavelength = 1450
# startIndex = np.where(rawData["Wavelength"]==startWavelength)[0]
# stopIndex = np.where(rawData["Wavelength"]==stopWavelength)[0] + 1
# x = rawData["Wavelength"][startIndex : stopIndex].values
# y = rawData["Intensity"][startIndex : stopIndex].values
#===============================================================================

#===============================================================================
# x = data[1:, 0]
# y = data[1:, 1]
#===============================================================================

#===============================================================================
# def gaussian(x, amp, cen, wid):
#     "1-d gaussian: gaussian(x, amp, cen, wid)"
#     return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))
#===============================================================================


def gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3, amp4, cen4, wid4, amp5, cen5, wid5, amp6, cen6, wid6 ):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp1/(sqrt(2*pi)*wid1)) * exp(-1 * (x-cen1)**2 /(2*wid1**2)) + \
        (amp2/(sqrt(2*pi)*wid2)) * exp(-1 * (x-cen2)**2 /(2*wid2**2)) + \
        (amp3/(sqrt(2*pi)*wid3)) * exp(-1 * (x-cen3)**2 /(2*wid3**2)) + \
        (amp4/(sqrt(2*pi)*wid4)) * exp(-1 * (x-cen4)**2 /(2*wid4**2)) + \
        (amp5/(sqrt(2*pi)*wid5)) * exp(-1 * (x-cen5)**2 /(2*wid5**2)) + \
        (amp6/(sqrt(2*pi)*wid6)) * exp(-1 * (x-cen6)**2 /(2*wid6**2))

gmod = Model(gaussian)
result = gmod.fit(y, x=x, amp1=5, amp2=5, amp3=5, amp4=5, amp5=5, amp6=5, \
                  cen1=830, cen2=950, cen3=1150, cen4=1310, cen5=1420, cen6=1530, \
                  wid1=1, wid2=1, wid3=1, wid4=1, wid5=1, wid6=1)

print(result.fit_report())

plt.plot(x, y,         'bo')
#plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.ylim(min(min(result.best_fit),min(x)),max(max(result.best_fit),max(y)))
plt.show()

print result.best_values