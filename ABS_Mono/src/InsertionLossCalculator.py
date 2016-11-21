'''
To calculate the insertion loss
two group data, the transmission without FUT and the transmission with FUT
Created on 21 Nov 2016

@author: Daytona
'''

import pandas as pd
import numpy as np

def combined_data(raw_data):
    print raw_data
    print raw_data.index
    
    start_wavelength = raw_data.index[0]
    stop_wavelength = raw_data.index[-1]
    wavelength_interval = raw_data.index[1] - raw_data.index[0]
    measurement_time = raw_data.shape[1]
    
    combined = []
    measurement_range = []
    for i in range(1, measurement_time+1):
        measurement_range.append(find_measurement_point(raw_data[i]))
    print measurement_range
    for wavelength in range(start_wavelength,stop_wavelength+wavelength_interval,wavelength_interval):
        pass
        #print wavelength
    
    # decied the size of the matrix
    #combined = raw_data
    #return combined
    
def find_measurement_point(data):
    result = 0
    for i in range(len(data)):
        if not np.isnan(data.iloc[i]):
            result += 1
    print result
    return result




if __name__ == '__main__':
    # raw data is from RawDataExtraction.py, so the data has not been combined
    # 1. read in SMF / FUT transmission spectra, short (~1000) and long (900~) range
    SMF_short_file = ""
    SMF_long_file = ""
    FUT_short_file = "../Data/FUT_short.csv"
    FUT_long_file = ""
    #raw_data_SMF_short = pd.read_csv()
    #raw_data_SMF_long = pd.read_csv()
    raw_data_FUT_short = pd.read_csv(FUT_short_file, header = None, index_col = 0, skiprows = 0)
    #raw_data_FUT_long = pd.read_csv()
    
    trans_combined_FUT_short = combined_data(raw_data_FUT_short)
    
    # 2. combined the transmission for long and short range for both data
    
    # 3. calculate the insertion loss for short and long range.
    
    # 4. rough combined the data and output the result