'''
To calculate the insertion loss
two group data, the transmission without FUT and the transmission with FUT
Created on 21 Nov 2016

@author: Daytona
'''

import pandas as pd
import numpy as np
import scipy.stats as ss

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
    # BEDF length
    length = 1
    # raw data is from RawDataExtraction.py, so the data has not been combined
    # 1. read in SMF / FUT transmission spectra, short (~1000) and long (900~) range
    SMF_short_file = "../Data/SMF_short.csv"
    SMF_long_file = "../Data/SMF_long.csv"
    FUT_short_file = "../Data/FUT_short.csv"
    FUT_long_file = "../Data/FUT_long.csv"
    
    raw_data_FUT_short = pd.read_csv(FUT_short_file, header = None, index_col = 0, skiprows = 0)
    raw_data_FUT_long = pd.read_csv(FUT_long_file, header = None, index_col = 0, skiprows = 0)
    raw_data_SMF_short = pd.read_csv(SMF_short_file, header = None, index_col = 0, skiprows = 0)
    raw_data_SMF_long = pd.read_csv(SMF_long_file, header = None, index_col = 0, skiprows = 0)
    #print raw_data_SMF_long
    
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
    combined_abs = []
    for i in range(result.shape[0]) :
        element = result["absorption_short"].tolist()[i]
        if not np.isnan(element) :
            combined_abs.append(element)
        else :
            combined_abs.append(result["absorption_long"].tolist()[i])
    result["combined_absorption"] = combined_abs
    
    result.to_csv("../Result/test.csv")
    
    
    #result = raw_data_FUT_short.join()
    print "done"   