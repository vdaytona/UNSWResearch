'''
Created on 26 Aug 2016
version 1. Used to combine the data obtained from one photo-detector 

@author: purewin7
'''
import pandas as pd
import time
import os


def run():
    path = "../Data/"
    file_list = os.listdir("../Data/")
    
    data = pd.read_csv(path + file_list[0], header = None, index_col = 0, skiprows = 0)
    
    for file_new in file_list[1:] :
        data_new = pd.read_csv(path + file_new, header = None, index_col = 0, skiprows = 0)
        data = pd.merge(data,data_new,left_index=True, right_index=True,how="outer")
        
    print data
    
    data.fillna(0)
    
    #print int(min(data.index))
    min_index = str(int(min(data.index)))
    max_index = str(int(max(data.index)))
    
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    fileName = '../Result/' + timeNow + '_MonoCombinedData_' + min_index + '-' + max_index + '.csv'
    print fileName
    
    data.to_csv(fileName,header = None)
    
    
if __name__ == '__main__':
    run()