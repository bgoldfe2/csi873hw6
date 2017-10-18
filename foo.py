# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:12:57 2017

@author: bruce
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math

def ReadInFiles(path,trnORtst):
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        print (fname)
        if fname.startswith(trnORtst):
            data = np.loadtxt(path + "\\" + fname)
            fullData.append(data)
    numFiles = len (fullData)
    print(numFiles)
   
    return fullData

def ReadInOneList(fullData,maxRows):
    allData = []
    numFiles = len (fullData)
    for j in range (numFiles):
        # allows for smaller data set sizes
        numRows = len (fullData[j])
        if (maxRows < numRows):
            numRows = maxRows
    
        for k in range(numRows):
            allData.append(fullData[j][k])
    return np.asarray(allData)

def ReadTrainingShort(path):
    train0 = np.loadtxt(path + "\\train0s.txt")
    train1 = np.loadtxt(path + "\\train1s.txt")
    cnt0 = len(train0)
    cnt1 = len(train1)
    print("counts are for zero " + str(cnt0) + " and one " + str(cnt1))
    my_data = np.asarray(train0)
    my_data[my_data > 0] = 1
    np.savetxt('fooout2.txt',my_data,fmt='%1i')
    tr0sums = my_data.sum(axis=0)
    print (tr0sums)
    np.savetxt('fooout.txt',tr0sums,fmt='%1i')
    
def MakeBinary(my_data):
    my_data[my_data > 0] = 1
    #HeatMap(my_data)
    #np.savetxt('fooout.txt',my_data,fmt='%1i')
    return my_data

def Train(trnData,index):
    freqList = np.zeros((10,784))
    start = 0
    end = index - 1
    # to iterate over the numbers 0-9 make range 0,9
    # this does a vertical count of each column to find
    # the frequency that a cell is '1' or written in
    for x in range(0,2):
        freqList[x,:] = trnData[start:end,1:785].sum(axis=0)
        start = start + index
        end = end + index
    HeatMap(freqList[0,:])
        
    return freqList

def testData(freqList,testList,trnNum):
    
    # read the first cell to find the number
    testItem = testList[0,:]
    print(testItem.shape)
    print('the number is ',testItem[0])
    
    # Get the individual cells out of the 784
    cellval = testList[0,1]
    print('the cell value is ',cellval)
    
    # apply an m-estimate probability
    # n is always the trnNum or number of training instances
    # this is the same for all giving a prior probability
    # of 1/10 or 0.1, so p=0.1
    
    # Get the frequency [number(0-9),cell(0-783)]
    # The cell value determines the nc, for cellval=1
    # nc = frequency for cellval=0 nc = trnNum-frequency
    if (cellval == 1):
        nc = freqList[0,0]
    else:
        nc = trnNum - freqList[0,0]
    n = trnNum
    p = 0.1
    m = 1  # picked 1 as a first try
    
    # formula from pg 179 in text for m-estimate probability
    m_est = (nc + m*p)/(n + m)
    print ('m estimate of probability is ',m_est)
    
    # The log probability will change the product of the 
    # probabilities into the sum of probabilities.  The 
    # probability will be calculated for each training set for
    # numbers 0-9. The maximum will be taken as the answer.
    # For the array use numpy log np.log function
    
    probNum0 = math.log(m_est)
    print('the log prob is ',probNum0)
    
def HeatMap(numberIn):
    #heat map to show numbers
    #plt.matshow(numberIn[0,1:785].reshape(28,28))
    plt.matshow(numberIn.reshape(28,28))
    plt.colorbar()
    plt.show()

def main():
    trnNum = 50
    tstNum = 50
    dpath = os.getcwd()+'\data3'
    #print (dpath)
    dataset = ReadInFiles(dpath,'train')
    #print(len(dataset))
    my_data = ReadInOneList(dataset,trnNum)
    
    np.savetxt('fooout.txt',my_data,fmt='%1i')
    
    #HeatMap(my_data)
    binData = MakeBinary(my_data)  
    
    freqArr = Train(binData,trnNum)
    np.savetxt('freqout.txt',freqArr,fmt='%1i')
    
    dataset2 = ReadInFiles(dpath,'test')
    #print(len(dataset))
    my_test = ReadInOneList(dataset2,trnNum)
    
    HeatMap(my_test[1,1:785])
    
    testData(freqArr,my_test,trnNum)
    #HeatMap(my_data)
    #np.savetxt('fooout2.txt',binData,fmt='%1i')
    
    
    
main()