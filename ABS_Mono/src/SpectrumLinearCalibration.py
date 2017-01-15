'''
Created on 3 Jan 2017
1. Some spectrum recorded in different time may vary in a linear relationship, like A(lamda) = f(lamda, B(lamda)), so we need to find this f to calibrate the spectra

@author: purewin7
'''

import pandas as pd
from scipy.stats import linregress



if __name__ == '__main__':
    # 1. read in data
    
    raw_data = pd.read_csv("../Data/Input.csv", header = None, index_col = 0, skiprows = 0)
    # print raw_data
    print raw_data.index
    
    
    # 2. set points for calculate the linear Relationship
    start_wavelength = 950
    stop_wavelength = 1000
    
    # 3. calculate the linear Relationship
    result = pd.DataFrame(index = raw_data.index)
    result["0"] = raw_data.iloc[:,0].values
    print result
    x_data = raw_data.iloc[:,0].loc[start_wavelength:stop_wavelength].values
    for x in range(len(raw_data.columns)-1) :
        y_data = raw_data.iloc[:,x+1].loc[start_wavelength:stop_wavelength].values
        slope, intercept, r_value, p_value, std_err = linregress(x_data,y_data)
        new_data = (raw_data.iloc[:,x+1].values - intercept) / slope
        result[str(x+1)] = new_data
    
    print result
    result.to_csv("../Result/test.csv",header=None)
    
    # 4. calculate the calibrated value according to the linear Equation
    
    # 5. output the result