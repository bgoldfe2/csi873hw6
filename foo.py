# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:12:57 2017

@author: bruce
"""
import os
import numpy as np
from itertools import islice

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
    print("counts are for zero " + str(cnt0) + " and one " + str(cnt1))
    my_data = np.asarray(train0)
    my_data[my_data > 0] = 1
    np.savetxt('fooout2.txt',my_data,fmt='%1i')
    tr0sums = my_data.sum(axis=0)
    print (tr0sums)
    np.savetxt('fooout.txt',tr0sums,fmt='%1i')
    
def ReadInSlice():
    path2 = 'C:\\Users\\bruce\\Documents\\GMU\\csi873\\hws\\hw6\\data2\\train0.txt'
    
    with open(path2) as myfile:
        head = list(islice(myfile, 2))
    print (head)
    
def Bayes():
    fooasdf = 2

def main():
    trnNum = 100
    tstNum = 50
    dpath = os.getcwd()+'\data3'
    #print (dpath)
    dataset = ReadInFiles(dpath)
    #print(len(dataset))
    my_data = ReadInOneList(dataset)
    print (my_data.shape)
    print (my_data.shape)
    my_data[my_data > 0] = 1
    print (my_data)
    np.savetxt('fooout.txt',my_data,fmt='%1i')
    
def main2():
    dpath = os.getcwd()+'\data3'
    print (dpath)
    #ReadTrainingShort(dpath)
    ReadInSlice()
    
    
main2()