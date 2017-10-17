# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:23:24 2017

@author: bruce
"""
import csv,glob
import os
import numpy as np

def readme(fname,skip_header=1):
    with open(fname) as f:
        for i in range(skip_header):
            f.next()
        return f.next()
    
def loadCsv3(path2data,datatype):
    pdata = path2data + '\\' + 'train*.txt'
    print (pdata)
    xx = glob.glob(pdata)
    print (xx)
    a = np.genfromtxt((readme(fname) for fname in glob.glob(pdata)))
    print('np shape is',a.shape)
    return a

def loadCsv2(path2data,datatype):
   
    alldata = []
    for i in os.listdir(path2data):
        if i.startswith(datatype):
            #files.append(path2data+open(i))
            pdata = path2data + '\\' + i
            with open(pdata, 'r') as f:
                reader = csv.reader(f)
                alldata.append(list(reader))
    
    my_data = np.array(alldata)
    print (my_data.shape)
    my_data[my_data > 0] = 1
    return my_data
  
def main():
    dpath = os.getcwd()+'\data'
    print (dpath)
    #filename = 'data/test0.txt'
    #splitRatio = 0.67
    dataset = loadCsv3(dpath,'train')
    print(dataset.max())
    #trainingSet, testSet = splitDataset(dataset, splitRatio)
    #print(('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet)))
    
main()