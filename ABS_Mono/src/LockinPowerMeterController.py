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
    print("start")
    time_start = time.time()
    # 1. link to lock-in
    rm = visa.ResourceManager()
    inst = rm.open_resource("GPIB0::23::INSTR")
    inst_pm = rm.open_resource("GPIB0::5::INSTR")
    
    time_interval = 30
    fiber_type = "P1202ARSLS-14s"
    irradiation_power = "40mW"
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    irradiation_time_expect = 60*60*24
    
    fileName = './' + timeNow + '_PhotoBleaching_' + fiber_type + '_' + irradiation_power + '.csv'
    irradiation_time = 0
    result = pd.DataFrame(columns = ['time','irradiation_time (sec)','intensity (V)','pump_power (W)','frequency (Hz)'])
    index = 0
    
    while irradiation_time < irradiation_time_expect:
        
        intensity = inst.query('Q1')

        try:
            power_pm = inst_pm.query('R_A?')
        except(visa.VisaIOError):
            power_pm = "0E-0"       


        #power_pm = inst_pm.query('R_A?')
        time_now = time.time()
        irradiation_time = time_now - time_start
        frequency = inst.query('f')
        #intensity = "112.45E-6\r\n"
        #frequency = "133\r\n"
        
        value = float(intensity.rsplit("E")[0])
        power = float(intensity.rsplit("E")[1])
        intensity = value / (np.power(10,-1*power))

        value_pm = float(power_pm.split("E")[0])
        power_pm = float(power_pm.split("E")[1])
        power_pm = value_pm / (np.power(10,-1*power_pm))

        frequency = float(frequency)
        
        strnow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result.loc[index] = [strnow,irradiation_time,intensity,power_pm,frequency]
        index += 1
        print result
        result.to_csv(fileName)
        if irradiation_time > 60*30 :
            time.sleep(60)
        else:
            time.sleep(time_interval)
        
    print "finished"