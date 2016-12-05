'''
Created on 15 Nov 2016
to fit the curve using power search method

@author: vdaytona
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score

# 1. read in data
rawData = pd.read_csv(".\Data\data.csv", header = None)
x_data = rawData[0].values
x_data = [float(x) for x in x_data]
y_data = rawData[1].values
#weight_data = rawData[2].values


def func1(x,a,b,c,d):
    return  a + b * np.exp(-np.power((x/c),d))

def square_score_func1(x,y,a,b,c,d):
    y_predict = []
    for x_value in x:
        y_predict.append(func1(x_value, a, b, c, d))
    square_score_fitted = r2_score(y,y_predict)
    #print square_score_fitted
    return square_score_fitted
    
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

if __name__ == '__main__':
    
    
    # 2. Preprocess if necessary
    
    # 3. set the function and search range and interval of parameter
    # function: a + b * exp(-(x/c)^d)
    a_start = 0.00025
    a_end = 0.00035
    a_interval = 0.000005
    a = [x for x in drange(a_start, a_end, a_interval)]
    b_start = 0.0004
    b_end = 0.0005
    b_interval = 0.000005
    b = [x for x in drange(b_start, b_end, b_interval)]
    c_start = 1500
    c_end = 3000
    c_interval = 20
    c = [x for x in drange(c_start, c_end, c_interval)]
    d_start = 0.34
    d_end = 0.44
    d_interval = 0.01
    d = [x for x in drange(d_start, d_end, d_interval)]
    
    # 4. calculate the best combination due to correlation coefficient
    loop = len(a) * len(b) * len(c) * len(d)
    print loop
    result = []
    loop_now = 0
    for a_value in a:
        for b_value in b:
            for c_value in c:
                for d_value in d:
                    r2_square_fiited = square_score_func1(x_data,y_data,a_value,b_value,c_value,d_value )
                    result.append((a_value,b_value,c_value,d_value,r2_square_fiited))
                    loop_now +=1
                    print (str(loop) + " in all, now is " +  str(loop_now) + " the r2_square is " +  str(r2_square_fiited) + " , finished " + str(float(loop_now) / float(loop) * 100) + " percent ")
                    
    result_sorted = sorted(result, key = lambda result : result[4],reverse=True)
    
    # 5. output the best 10000 result and all result
    result_pd = pd.DataFrame(data=result_sorted)
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    fileName_best = './Result/' + timeNow + '_FittingPowerSearch_best' + '.csv'
    fileName_all = './Result/' + timeNow + '_FittingPowerSearch' + '.csv'
    result_pd[0:20000].to_csv(fileName_best,header = None)
    #result_pd.to_csv(fileName_all,header = None)
    #result_pd = pd.to_csv
    
    # 6. Draw the best result of fitting
    #y_fit = []
    #for x_value in x_data:
    #    y_fit.append(func1(x_value, result_sorted[0][0], result_sorted[0][1], result_sorted[0][2], result_sorted[0][3]))
    #print result_sorted[0]
    #print y_fit
    #print y_data
    #plt.plot(x_data, y_data)
    #plt.plot(x_data,y_fit)
    #plt.show()