'''
Created on 24 Nov 2016

@author: Daytona
'''
import visa
import time
from datetime import datetime
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 1. link to lock-in
    rm = visa.ResourceManager()
    inst = rm.open_resource("GPIB0::23::INSTR")
    
    time_interval = 20
    fiber_type = "P1202ARSLS-14s"
    irradiation_power = "40mW"
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    irradiation_time_expect = 80
    fileName = './' + timeNow + '_PhotoBleaching_' + fiber_type + '_' + irradiation_power + '.csv'
    time_start = time.time()
    irradiation_time = 0
    result = pd.DataFrame(columns = ['time','irradiation_time','intensity','frequency'])
    index = 0
    
    while irradiation_time < irradiation_time_expect:
        
        time_now = time.time()
        irradiation_time = time_now - time_start
        
        #intensity = 1.0
        #frequency = 133
        
        intensity = inst.query('Q1')
        frequency = inst.query('f')
        strnow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result.loc[index] = [strnow,irradiation_time,intensity,frequency]
        index += 1
        print result
        result.to_csv(fileName)
        time.sleep(time_interval)
        
    print "finished"