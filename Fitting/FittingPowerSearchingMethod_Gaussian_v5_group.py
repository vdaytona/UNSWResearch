'''
Created on 15 Nov 2016
to fit the curve using power search method
v2: added correlation coefficient
v3: added weight calculate to r2 score
v4: added weight choice
v5: change to Gaussian fitting

@author: vdaytona
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score
from time import sleep
from __builtin__ import file
from numpy import sqrt, pi, exp, linspace, loadtxt

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

def gaussian_func(x,amp,cen,wid):
    sta_dev = 2.3548 / wid
    return (amp/(sqrt(2*pi)*sta_dev)) * exp(-(x-cen)**2 /(2*sta_dev**2))

def multi_gaussian_func(x, amp_para, cen_para, wid_para):
    predict = []
    for i in range(len(amp_para)):
        predict = predict + gaussian_func(x, amp_para[i], cen_para[i], wid_para[i])
    return predict
        
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
        
def index_list_generator(list):
    len_list = [len(x) for x in list]
    index_list = []
    for i0 in range(len_list[0]):
        for i1 in range(len_list[1]):
            for i2 in range(len_list[2]):
                for i3 in range(len_list[3]):
                    for i4 in range(len_list[4]):
                        index_list.append(i0,i1,i2 ,i3,i4)
    return  index_list

def para_find(para_list, index_list):
    result = []
    for i in range(5):
        result.append(para_list[i][index_list[i]])
    return result
    

def start_fitting(x_data, y_data, weight_data, amp,cen, wid,output_file_name):    
    
    # set the function and search range and interval of parameter
    # function: a + b * exp(-(x/c)^d)
    amp_sep_number = 50.0
    amp_range_list = []
    for i in range(len(amp)):
        start = amp[i][0]
        end = amp[i][1]
        amp_interval = (start  - end) / amp_sep_number
        amp = [x for x in drange(start, end, amp_interval)]
        amp_range_list.append(amp)
        
    cen_range_list = []
    for i in range(len(cen)):
        start = cen[i][0]
        end = cen[i][1]
        interval = 2.0
        cen_range_list.append([x for x in drange(start, end, interval)])
        
    wid_range_list = []
    for i in range(len(wid)):
        start = wid[i][0]
        end = wid[i][1]
        interval = 2.0
        wid_range_list.append([x for x in drange(start, end, interval)])
    
    # 4. calculate the best combination due to correlation coefficient
    loop = (len(amp_range_list[0]) * len(cen_range_list[0]) * len(wid_range_list[0]))^5
    print loop
    result = []
    loop_now = 0
    amp_index_list = index_list_generator(amp_range_list)
    cen_index_list = index_list_generator(cen_range_list)
    wid_index_list = index_list_generator(wid_range_list)
    
    for amp_index in amp_index_list:
        for cen_index in cen_index_list:
            for wid_index in wid_index_list:
                amp_para = para_find(amp_range_list, amp_index)
                cen_para = para_find(cen_range_list, cen_index)
                wid_para = para_find(wid_range_list, wid_index)
                predict_y = multi_gaussian_func(x, amp_para, cen_para, wid_para)
                #####
                
                
                            
                        

    
    for a_value in amp:
        for c_value in cen:
            for d_value in wid:
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
    
def substract_background(amp,background):
    new_amp = []
    for list in amp:
        new_amp.append([x - background for x in list])
    return new_amp

if __name__ == '__main__':
    global weight_if
    weight_if = False
    # file list
    file_list = ['data1']
    
    output_list = ['Gaussian_fitting_20s']
    
    background = 0.000473182
    amp = [ [0.001,0.001/2.0],[0.003,0.0015],[0.001,0.001/2.0],[0.00175,0.00175/2.0],[0.00334,0.00334/2.0]]
    cen = [[962.0,974.],[1086.,1096.],[1348.,1358.],[1414.,1422.],[1534.,1540.]]
    wid = [[30.,90.],[110.,160.],[50.,100.],[80.,150.],[26.,40.]]
    amp = substract_background(amp, background)
    
    print range(len(file_list))
    for f in range(len(file_list)):
        file_name = './Data/' + file_list[f] + '.csv'
        timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        if weight_if == True:
            output_file_name = './Result/' + output_list[f] +  timeNow + '_GroupFittingGrid_Gaussian_weighted_v5_' + '.csv'
        else :
            output_file_name = './Result/' + output_list[f] +  timeNow + '_GroupFittingGrid_Gaussian_Noweighted_v5_' + '.csv'
        
        rawData = pd.read_csv(file_name, header = None)
        x_data = rawData[0]
        x_data = [float(x) for x in x_data]
        y_data = rawData[1]
        # create weight list, no use at moment
        weight_list = []
        for i in range(len(y_data)):
            if i < len(y_data)-1 :
                weight_list.append(np.power((abs(y_data[i] - y_data[i+1])),1))
            else:
                weight_list.append(weight_list[-1])
        start_fitting(x_data, y_data, weight_list, amp, cen, wid, output_file_name)