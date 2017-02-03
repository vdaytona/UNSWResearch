'''
Created on 15 Nov 2016
to fit the curve using power search method
v2: added correlation coefficient
v3: added weight calculate to r2 score

@author: vdaytona
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score
from time import sleep
from __builtin__ import file

#===============================================================================
# # 1. read in data
# rawData = pd.read_csv(".\Data\data.csv", header = None)
# x_data = rawData[0].values
# x_data = [float(x) for x in x_data]
# y_data = rawData[1].values
# 
# # create weight list
# weight_list = []
# for i in range(len(y_data)):
#     if i < len(y_data)-1 :
#         weight_list.append(abs(y_data[i] - y_data[i+1]))
#     else:
#         weight_list.append(weight_list[-1])
# print len(weight_list)
# print len(y_data)
#===============================================================================
weight_if = True

def func1(x,a,c,d):
    return  a + (1-a) * np.exp(-np.power((x/c),d))

def square_score_func1(x,y,a,c,d,weight_list):
    y_predict = []
    for x_value in x:
        y_predict.append(func1(x_value, a, c, d))
    #print weight_if
    if weight_if == True:
        square_score_fitted = r2_score(y,y_predict, sample_weight = weight_list)
    else :
        square_score_fitted = r2_score(y,y_predict)
    return square_score_fitted

def cor_coef_func1(x,y,a,c,d):
    y_predict = []
    for x_value in x:
        y_predict.append(func1(x_value, a, c, d))
    cor_coeff_result = np.corrcoef(y,y_predict)[0][1]
    return cor_coeff_result
    
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def start_fitting(x_data, y_data, weight_data, a,c,output_file_name):    
    
    # set the function and search range and interval of parameter
    # function: a + b * exp(-(x/c)^d)
    a_start = a[0]
    a_end = a[1]
    a_interval = 0.002
    a = [x for x in drange(a_start, a_end, a_interval)]
    #b_start = b[0]
    #b_end = b[1]
    #b_interval = 0.005
    #b = [x for x in drange(b_start, b_end, b_interval)]
    c_start = c[0]
    c_end = c[1]
    c_interval = 20
    c = [x for x in drange(c_start, c_end, c_interval)]
    d_start = 0.30
    d_end = 0.45
    d_interval = 0.002
    d = [x for x in drange(d_start, d_end, d_interval)]
    
    # 4. calculate the best combination due to correlation coefficient
    loop = len(a) * len(c) * len(d)
    print loop
    result = []
    loop_now = 0
    for a_value in a:
        #for b_value in b:
        for c_value in c:
            for d_value in d:
                r2_square_fiited = square_score_func1(x_data,y_data,a_value,c_value,d_value,weight_list)
                corr_result = cor_coef_func1(x_data,y_data,a_value,c_value,d_value )
                total_score = corr_result + r2_square_fiited
                first_value_diff = abs(a_value + (1-a_value) * np.exp(-np.power((x_data[0]/c_value),d_value)) - y_data[0])
                result.append((a_value,c_value,d_value,r2_square_fiited,corr_result,total_score,first_value_diff))
                loop_now +=1
                print (str(loop) + " in all, now is " +  str(loop_now) + " the r2_square + corr_score is " +  str(total_score) + " , finished " + str(float(loop_now) / float(loop) * 100) + " percent ")
                    
    result_sorted = sorted(result, key = lambda result : result[3],reverse=True)
     
    
    # 5. output the best 10000 result and all result
    result_pd = pd.DataFrame(data=result_sorted)
    result_pd[0:50000].to_csv(output_file_name,header = None)
    print "best r2 score"
    print sorted(result, key = lambda result : result[3],reverse=True)[0]
    print "best corr score"
    print sorted(result, key = lambda result : result[4],reverse=True)[0]
    print "best total score"
    print sorted(result, key = lambda result : result[5],reverse=True)[0]
    print "best first value"
    print sorted(result, key = lambda result : result[6],reverse=False)[0]

if __name__ == '__main__':
    global weight_if
    weight_if = False
    # file list
    file_list = ['data1']
    
    output_list = ['2017-02-02-12-21-F151108-11s_150mA_1cm_']
    
    a = [ [0.40,0.6]]
    #b = [[0.45,0.55], [0.4,0.6], [0.60,0.62]]
    c = [ [1000,10000]]
    print range(len(file_list))
    for f in range(len(file_list)):
        file_name = './Data/' + file_list[f] + '.csv'
        timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        if weight_if == True:
            output_file_name = './Result/' + output_list[f] +  timeNow + '_GroupFittingGrid_weighted_v4_' + '.csv'
        else :
            output_file_name = './Result/' + timeNow + '_GroupFittingGrid_Noweighted_v4_' + output_list[f] + '.csv'
        
        rawData = pd.read_csv(file_name, header = None)
        x_data = rawData[0]
        x_data = [float(x) for x in x_data]
        y_data = rawData[1]
        # create weight list
        weight_list = []
        for i in range(len(y_data)):
            if i < len(y_data)-1 :
                weight_list.append(np.power((abs(y_data[i] - y_data[i+1])),1))
            else:
                weight_list.append(weight_list[-1])
        start_fitting(x_data, y_data, weight_list, a[f], c[f], output_file_name)