# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:12:57 2017

@author: bruce
"""
import os
import numpy as np

def ReadInFiles(path):
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        data = np.loadtxt(path + "\\" + fname)
        fullData.append(data)
    numFiles = len (fullData)
    print(numFiles)
   
    return fullData

def ReadInOneList(fullData):
    allData = []
    numFiles = len (fullData)
    for j in range (numFiles):
        numRows = len (fullData[j])
    
        for k in range(numRows):
            allData.append(fullData[j][k])
    return np.asarray(allData)

def ReadTrainingShort(path):
    train0 = np.loadtxt(path + "\\train0s.txt")
    train1 = np.loadtxt(path + "\\train1s.txt")
    cnt0 = len(train0)
    cnt1 = len(train1)
    print("counts are for zero" + cnt0 + "and one" +cnt1)

def main():
    dpath = os.getcwd()+'\data3'
    print (dpath)
    dataset = ReadInFiles(dpath)
    print(len(dataset))
    my_data = ReadInOneList(dataset)
    print (my_data.shape)
    print (my_data.shape)
    my_data[my_data > 0] = 1
    print (my_data)
    np.savetxt('fooout.txt',my_data,fmt='%1i')
    
def main2():
    dpath = os.getcwd()+'\data3'
    print (dpath)
    ReadTrainingShort(dpath)
    
    
main()