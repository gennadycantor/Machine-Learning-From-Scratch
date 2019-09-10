# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:01:55 2019

@author: miru
"""

import numpy as np
import pandas as pd
import math

def preprocess(fname):
    data = pd.read_csv(fname)
    data = np.array(data)
    return data

def trainTestsplit(ratio,X,Y):
    cutoff = int(math.floor(ratio * len(Y)))
    trainX = X[:cutoff,:]
    testX = X[cutoff:,:]
    trainY = Y[:cutoff]
    testY = Y[cutoff:]
    return trainX, testX, trainY, testY

def signFunc(Z):
    
    return np.sign(Z)
    
"""
this function evaluates weights on a new testset 
"""
def evaluate(weights, bias, testX, testY):
    Z = np.matmul(weights, testX.T) + bias
    
    output = signFunc(Z)
    
    output = output[0].tolist()
   
    counter = 0
    for i in range(len(output)):
        if output[i] != testY[i]:
            counter +=1
            #print(counter)
            #print(counter)
    error = ( counter / float(len(testY)) ) * 100
    return counter, error

def evaluateThree(weights, bias, trainX, trainY):
    Z = np.matmul(weights, trainX.T) + bias
    
    output = signFunc(Z)
   
    output = output.tolist()
   
    counter = 0
    for i in range(len(output)):
        if output[i] != trainY[i]:
            counter +=1
           
    error = ( counter / float(len(trainY)) ) * 100
    return counter, error
    
def computeCost(A):
    counter = 0
  
    for i in range(len(A[0])):
        if A[0][i] != 0:
            counter+=1
    ratio = counter / float(len(A[0]))
    ratio = ratio*100
    return ratio

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
    print("For " + str(j) + " th hold out set, the cost per epoch is the following: " + "\n")
    w, b, cost = trainWeights(testSetX,testSetY, 1000, 0.01, 0.1)
    counter, error = evaluate(w,b,holdoutX, holdoutY)
    subsetY.insert(j, garbage1)
    subsetX.insert(j, garbage2)
    #return testSetX, holdoutX, testSetY, holdoutY, error
    return error

"""
extracts top three features based on each components magnitude
we see that 1st, 3rd, and last are the biggest
"""
def extractFeatures(Weights):
    #print(Weights[0][1])
    newWeights = np.zeros(len(Weights[0]))
    newWeights[0] = Weights[0][0]
    newWeights[2] = Weights[0][2]
    newWeights[len(Weights[0]) - 1] = Weights[0][len(Weights[0]) - 1]
    return newWeights

def trainWeights(trainX, trainY, numitr, alpha, epsilon):
    """ 
    We will initiazlise weights and bias to 1.
    Then iteratively, we will compute W.Tx and see how many
    of the points in the training set are being misclassified.
    We will than compute the cost 
    We will then pick a random point among misclassified points, and
    we will simply rotate our linear classifier boundary towards that point
    we will repeat this until our sequence of costs achieve caucy convergence
    """
   
    weight = np.ones(trainX.shape[1])
    bias = np.zeros(1)
    cost = []
    #epsilonCost = []
    # we will vectorize this, we dont want to see nested for loops
    for i in range(numitr):
        # in pyton, by default vector is a row vector
        Z = np.matmul(weight, trainX.T) + bias
        # now we have 1xm vector, m is the number of training examples
        # apply sign function 
        output = signFunc(Z)
        #print("output", output.shape)
        
        # we only want to rotate if we missclassified
        A = trainY.T - output
        #print("A's shape", A.shape)
        costPer = computeCost(A)
        cost.append(costPer)
        #epsilonCost.append(costPer)
        if i % 200 == 0:
            cost.append(costPer)
            print("@ Epoch " + str(i/100) + " cost is: " + str(cost[i/100]))
        
        # now rotate towards missclassified point in the feature direction
        dW = (np.matmul(A, trainX))
        #print(dW.shape)
        dB = np.sum(A)
        # now update weight
        weight = weight + alpha*dW
        bias += alpha*dB
    print("\n")   
    return weight, bias, cost
        


    

def main():
    X = preprocess('face_emotion_data_X.csv')
    Y = preprocess('face_emotion_data_y.csv')
    trainX, testX, trainY, testY = trainTestsplit(0.7, X, Y)
    # store weight vector and bias
    weight, b, cost = trainWeights(trainX, trainY, 1000, 0.01, 0.1)
    
    print(weight)
    print(b)
    # store number of missclassifications
    missClassifications, error = evaluate(weight, b, testX, testY)
    # get new Weights 3 weights
    newWeights = extractFeatures(weight)
    print(newWeights)
    # now fit to whole training set
    missClassificationsThree, errorThree = evaluateThree(newWeights, b, trainX, trainY)
    
    subsetX, subsetY = crossValid(X, Y)
    
    #initialize error list
    errorlist = []
    for j in range(8):
        error = evalcrossValid(subsetX, subsetY, j)
        errorlist.append((error))
    print(errorlist)
    print("\n")
    
    avgError = sum(errorlist) / float(len(errorlist))
    print("Final average Error is: " + str(avgError) + " %")
    
    
    
main()