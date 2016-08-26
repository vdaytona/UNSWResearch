#!/usr/bin/python3
# substract data from sequence OSA files for small signal data process
# This is an exercise file from Python 3
# @Mingjie DING

import linecache
import math
import time
from linecache import getlines

def main():
    StartWavelength = 700.0
    EndWavelength = 1700.0
    SamplePoint = 1001
    StartRow = 32
    EndRow = 1033
    StartFile = 0
    EndFile = 35
    outputData = []
    
    for x in range(StartFile, EndFile+1):
        data = []
        fileName = ('./TR_000%02d.csv' % (x))
        lines = [line.strip() for line in open(fileName)]
        for y in range(StartRow,EndRow):
            #print(lines[y].split(",")[2])
            PowerWithNoisemW = math.pow(10,float(lines[y].split(",")[1])/10)
            NoisemW = math.pow(10,float(lines[y].split(",")[2])/10)
            NoiseReducedPowermW = PowerWithNoisemW - NoisemW
            data.append(NoiseReducedPowermW)
        outputData.append(data)
    
    
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    fileName = './SmallSignalResult-' + timeNow + '.csv'
    f = open(fileName,'w')
    for x in range (0,len(outputData[0])):
        wavelength = StartWavelength + (EndWavelength-StartWavelength) * (x+1-1)/(SamplePoint-1)
        inputString = str(wavelength)+ ","
        for y in range (0,len(outputData)):
            if y != len(outputData):
                inputString += str(outputData[y][x]) + ","
            else:
                inputString += str(outputData[y][x])
        inputString += "\n"
        #print(inputString)
        f.write(inputString)
    
    f.close()
    print("The result file is " + fileName)
    
    
        #with open(fileName, 'r') as file:
            #for y in range(33,35):
                #print(file.readlines(y))
            
        

#with open('/Users/Daytona/Dropbox/Research_UNSW/Experiment/PowerMeterCalibrationByOSAInLowInputPower/20150516/Summay.CSV', 'wb') as csvfile

if __name__ == "__main__": main()
