'''
Combined the function of RawDataExtraction.py and InsetionLossCalculator
Only need to input the fileName
Created on 28 Nov 2016

@author: Daytona
'''
import pandas as pd
import numpy as np
import scipy.stats as ss

def rawDataExtraction(fileList):
    path = '../Data/'
    fileList = [path + x for x in fileList]
    print fileList
        
    data = pd.read_csv(path + fileList[0], header = None, index_col = 0, skiprows = 0)
    
    if len(fileList) >= 1 :
        for file_new in fileList[1:] :
            data_new = pd.read_csv(path + file_new, header = None, index_col = 0, skiprows = 0)
            data = pd.merge(data,data_new,left_index=True, right_index=True,how="outer")
    data.fillna(0)
    print data.shape[1]
    column_name = range(data.shape[1])
    column_name = [x + 1 for x in column_name]
    print column_name
    data.columns = column_name
    print data
    return data

def combined_data(raw_data):
    
    #===========================================================================
    # start_wavelength = raw_data.index[0]
    # stop_wavelength = raw_data.index[-1]
    # wavelength_interval = raw_data.index[1] - raw_data.index[0]
    #===========================================================================
    
    measurement_time = raw_data.shape[1]
    
    combined_data_list = []
    measured_point = []
    for i in range(1, measurement_time+1):
        measured_point.append(find_measurement_point(raw_data[i]))
        
    # get rank
    rank = ss.rankdata(measured_point, method='ordinal')-1
    
    ########### modified 2016/11/26 at UNSW
    for i in range(raw_data.shape[0]):
        rank_no = 0
        while rank_no < len(rank) :
            position = int(np.where(rank == rank_no)[0])
            element = raw_data.iloc[i].tolist()[position]
            if not np.isnan(element) :
                combined_data_list.append(element)
                break
            rank_no += 1
    raw_data["combined"] = combined_data_list
    return raw_data

def find_measurement_point(data):
    result = 0
    for i in range(len(data)):
        if not np.isnan(data.iloc[i]):
            result += 1
    return result

if __name__ == '__main__':
    
    length = 1.0
    file_name_pre = "161123"
    SMF_short_file = ["26","27"]
    SMF_long_file = ["25"]
    FUT_short_file = ["20","21","22","23"]
    FUT_long_file = ["24"]
    SMF_short_file = [file_name_pre + x + ".csv" for x in SMF_short_file]
    SMF_long_file = [file_name_pre + x + ".csv" for x in SMF_long_file]
    FUT_short_file = [file_name_pre + x + ".csv" for x in FUT_short_file]
    FUT_long_file = [file_name_pre + x + ".csv" for x in FUT_long_file]
    
    raw_data_FUT_short = rawDataExtraction(FUT_short_file)
    raw_data_FUT_long = rawDataExtraction(FUT_long_file)
    raw_data_SMF_short = rawDataExtraction(SMF_short_file)
    raw_data_SMF_long = rawDataExtraction(SMF_long_file)
    print raw_data_FUT_short
    
    # 2. combined the transmission for long and short range for both data
    print "combine FUT short"
    raw_data_FUT_short = combined_data(raw_data_FUT_short)
    print "combine FUT long"
    raw_data_FUT_long = combined_data(raw_data_FUT_long)
    print "combine SMF short"
    raw_data_SMF_short = combined_data(raw_data_SMF_short)
    print "combine SMF long"
    raw_data_SMF_long = combined_data(raw_data_SMF_long)
    #print raw_data_SMF_long
    
    
    print "combined done"
    
    # 3. calculate the insertion loss for short and long range.
    raw_data_FUT_short = raw_data_FUT_short.join(raw_data_SMF_short,lsuffix='_FUT_short', rsuffix = '_SMF_short')
    raw_data_FUT_short["absorption_short"] = 10 * np.log10(raw_data_FUT_short["combined_SMF_short"] / raw_data_FUT_short["combined_FUT_short"]) / length
    raw_data_FUT_short.index.name = "wavelength"
    
    raw_data_FUT_long = raw_data_FUT_long.join(raw_data_SMF_long,lsuffix='_FUT_long', rsuffix = '_SMF_long')
    raw_data_FUT_long["absorption_long"] = 10 * np.log10(raw_data_FUT_long["combined_SMF_long"] / raw_data_FUT_long["combined_FUT_long"]) / length
    raw_data_FUT_long.index.name = "wavelength"
    
    #print raw_data_FUT_long
    
    
    
    # 4. rough combined the data and output the result
    result = pd.merge(raw_data_FUT_short,raw_data_FUT_long,left_index=True, right_index=True,how="outer")
    # move the short absorption spectrum to meet the difference before long and short region
    interval_long_short = result.loc[1000]["absorption_long"] - result.loc[1000]["absorption_short"]
    
    combined_abs = []
    for i in range(result.shape[0]) :
        element = result["absorption_short"].tolist()[i]
        if not np.isnan(element) :
            combined_abs.append(element + interval_long_short)
        else :
            combined_abs.append(result["absorption_long"].tolist()[i])
    result["combined_absorption"] = combined_abs
    
    result.to_csv("../Result/test.csv")
    
    
    #result = raw_data_FUT_short.join()
    print "done"   
    