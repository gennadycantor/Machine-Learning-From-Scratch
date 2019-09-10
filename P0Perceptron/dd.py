# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 03:45:23 2019

@author: miru
"""

import numpy as np
import pandas as pd
import math

def preprocess(fname):
    data = pd.read_csv(fname)
    data = np.array(data)
    return data

def lsq(X, t1):
    W = np.linalg.lstsq(X, t1, rcond=None)[0]
    return W


def signFunc(Z):
   
    return np.sign(Z)

def evaluateThree(weights, trainX, trainY):
    Z = np.matmul(weights.T, trainX.T)
    
    output = signFunc(Z)
   
    output = output.tolist()

    counter = 0
    for i in range(len(output)):
        if output[i] != trainY[i]:
            counter +=1
            #print(counter)
            #print(counter)
    error = ( counter / float(len(trainY)) ) * 100
    return counter, error

def extractFeatures(Weights):
   
    newWeights = np.zeros(len(Weights))
    newWeights[0] = Weights[1]
    newWeights[2] = Weights[2]
    newWeights[3] = Weights[3]
    return newWeights

def evaluate(weights, testX, testY):
    Z = np.matmul(weights.T, testX.T)
    
    output = signFunc(Z)
   
    output = output[0].tolist()
    
    counter = 0
    for i in range(len(output)):
        if output[i] != testY[i]:
            counter +=1
            
    error = ( counter / float(len(testY)) ) * 100
    return counter, error

def crossValid(X, Y):
    subsetX = []
    subsetY = []
    numInstances = X.shape[0]
    index = numInstances / 16
    for i in range(index+1):
        subsetX.append(X[i*16:(i+1)*16,:])
        subsetY.append(Y[i*16:(i+1)*16])
    return subsetX, subsetY

def evalcrossValid(subsetX, subsetY,j):
    error = 0
    for i in range(len(subsetX)):
        if i == j:
            holdoutX = subsetX[i]
            holdoutY = subsetY[i]
    
    garbage1 = subsetY.pop(j)
    garbage2 = subsetX.pop(j)
    # now concat
    testSetX = np.vstack(subsetX)
    testSetY = np.vstack(subsetY)
    w = lsq(testSetX,testSetY)
    counter, error = evaluate(w, holdoutX, holdoutY)
    subsetY.insert(j, garbage1)
    subsetX.insert(j, garbage2)
    #return testSetX, holdoutX, testSetY, holdoutY, error
    return error
    
def main():
    X = preprocess('face_emotion_data_X.csv')
    Y = preprocess('face_emotion_data_y.csv')
    w = lsq(X, Y)
    missclassification, error = evaluate(w, X, Y)
    print("Initial weight vector is: ",w)
    print("\n")
   #print(type(w))
    
    print("we have this many missclassified points: " + str(missclassification)
        + "  .And the error is: " + str(error) + " %" + "\n")
    newW = extractFeatures(w)
    
    print("Modified 3 component weight vector is: ", newW)
    print("\n")
    
    newMiss,newError = evaluateThree(newW, X,Y)
    print("we have this many missclassified points with most important 3 components: " + str(newMiss)
        + " .And the error is: " + str(newError) + " %" + "\n")
    
    subsetX, subsetY = crossValid(X, Y)
    
    
    #initialize error list
    errorlist = []
    for j in range(8):
        error = evalcrossValid(subsetX, subsetY, j)
        errorlist.append((error))
    print("List of holdoutset error rates: ",errorlist)
    print("\n")
    
    avgError = sum(errorlist) / float(len(errorlist))
    print("Final average Error is: " + str(avgError) + " %")
    

    
    
main()