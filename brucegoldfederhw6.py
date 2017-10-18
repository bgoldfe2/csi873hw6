# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:06:45 2017

@author:    Bruce Goldfeder
            CSI 873
            Fall 2017
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math

def ReadInFiles(path,trnORtst):
    # This reads in all the files from a directory filtering on what the file
    # starts with
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        if fname.startswith(trnORtst):
            print (fname)
            data = np.loadtxt(path + "\\" + fname)
            fullData.append(data)
    numFiles = len (fullData)
    print(numFiles)
   
    return fullData

def ReadInOneList(fullData,maxRows):
    # This function combines all of the data into one array for ease of use
    # It contains a capping ability to configure how many results to use
    allData = []
    numFiles = len (fullData)
    for j in range (numFiles):
        # allows for smaller data set sizes
        numRows = len (fullData[j])
        #print('numrows,maxrows ',numRows,maxRows)
        if (maxRows < numRows):
            numRows = maxRows
    
        for k in range(numRows):
            allData.append(fullData[j][k])
    return np.asarray(allData)

def MakeBinary(my_data):
    # This module converts the 0-255 to 0 or 1 binomial
    my_data[my_data > 0] = 1
    
    return my_data

def Train(trnData,index):
    # The frequency list will contain a count of the frequency for each cell
    # in the dataset that has writing in it
    freqList = np.zeros((10,784))
    start = 0
    end = index - 1
    # to iterate over the numbers 0-9 make range 0,9
    # this does a vertical count of each column to find
    # the frequency that a cell is '1' or written in
    for x in range(0,10):
        freqList[x,:] = trnData[start:end,1:785].sum(axis=0)
        start = start + index
        end = end + index
    # provides a visualization to check functionality of the run
    HeatMap(freqList[0,:])
        
    return freqList

def testData(freqList,testList,trnNum,tstNum):
    
    # Make an array to hold the probabilities for a test sample
    probList = np.zeros((10,784))
    
    # Make an array to hold the accuracy figures
    numT = tstNum*10
    accuracyList = np.zeros((numT,3))
    
    
    # Loop through the number of test samples * 10 numbers
    for testNum in range(0,numT):
    
        # read the first cell to find the number
        testItem = testList[testNum,:]
        #HeatMap(testItem[1:785])
        #print(testItem.shape)
        accuracyList[testNum,0] = testItem[0]
        #print('the number is ',testItem[0])
        
        # Loop over all the number frequency sets 0-9
        for trainSet in range(0,10):
        
            # Loop over all the cell values for the test sample
            for cell in range(1,785):
                
                # Get the individual cells out of the 784 [test,cell]
                cellVal = testItem[cell]
                #print('the cell value is ',cellval)
                
                # Get the corresponding frequency value
                freqVal = freqList[trainSet,cell-1]
                
                # apply an m-estimate probability
                # n is always the trnNum or number of training instances
                # this is the same for all giving a prior probability
                # of 1/10 or 0.1, so p=0.1
                
                # Get the frequency [number(0-9),cell(0-783)]
                # The cell value determines the nc, for cellval>0
                # nc = frequency for cellval=0 nc = trnNum-frequency
                if cellVal > 0:
                    nc = freqVal
                else:
                    nc = trnNum - freqVal
                n = trnNum
                p = 0.1
                m = 1  # picked 1 as a first try
                
                # formula from pg 179 in text for m-estimate probability
                m_est = (nc + m*p)/(n + m)
                #print ('m estimate of probability is ',m_est)
                
                # The log probability will change the product of the 
                # probabilities into the sum of probabilities.  The 
                # probability will be calculated for each training set for
                # numbers 0-9. The maximum will be taken as the answer.
                # For the array use numpy log np.log function
                logVal = math.log(m_est)
                probList[trainSet,cell-1] = logVal
                #print('the cellVal, freqVal, nc, logVal ',cellVal, freqVal,nc, logVal )
        #print ('shape of probList ',probList.shape)
        sumForTest0 = probList.sum(axis=1)
       
        #print ('the sum of probs is ',sumForTest0)
        winnerWinnerChickenDinner = sumForTest0.argmax()
        accuracyList[testNum,1] = winnerWinnerChickenDinner
        # End the individual test
    
    # If both the input and the algorithm are the same put a 1 in third column
    accuracyList[:,2][accuracyList[:,0] - accuracyList[:,1] == 0] = 1
    return accuracyList

def Output(resultList,index):
    # converts the test results into counts per number (10 counts for 0-9)
    outList = np.zeros(10)
    start = 0
    end = index - 1
    # to iterate over the numbers 0-9 make range 0,9
    # this does a vertical count of each column to find
    # the frequency that a cell is '1' or written in
    for x in range(0,10):
        outList[x] = resultList[start:end,2].sum(axis=0)
        start = start + index
        end = end + index
        
    return outList
        
def HeatMap(numberIn):
    #heat map to show numbers
    plt.matshow(numberIn.reshape(28,28))
    plt.colorbar()
    plt.show()

def main():
    # Theses are the number counts for training and test data sets
    trnNum = 5000
    tstNum = 890
    
    dpath = os.getcwd()+'\data'
    
    # Read in the Training data first
    dataset = ReadInFiles(dpath,'train')
    my_data = ReadInOneList(dataset,trnNum)
    #np.savetxt('fooout.txt',my_data,fmt='%1i')
    
    # Convert the 0-255 to 0 or 1 values in data
    binData = MakeBinary(my_data)  
    
    # Create the frequency array for the training data
    freqArr = Train(binData,trnNum)
    #np.savetxt('freqout.txt',freqArr,fmt='%1i')
    
    # Read in the test data
    dataset2 = ReadInFiles(dpath,'test')
    my_test = ReadInOneList(dataset2,tstNum)
    
    # Test the data using the training set
    results = testData(freqArr,my_test,trnNum,tstNum)
    
    
    #results[:,2][results[:,0] - results[:,1] == 0] = 1
    
    #np.savetxt('fooout3.txt',results,fmt='%1i')
    
    outputs = Output(results,tstNum)
    print(outputs)
    print(outputs.sum(axis=0))
    
    # print the percentages per number tested
    print(outputs/tstNum)
    
    # print the percentage correct over all the numbers tested
    print (outputs.sum(axis=0)/(tstNum*10))
    
main()