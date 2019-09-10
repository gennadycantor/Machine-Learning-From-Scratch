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
    for i in range(len(Z)):
        
        if Z[i] > 0:
            Z[i] = 1
        elif Z[i] < 0:
            Z[i] = -1
    return Z
    

def evaluate():
    return 0
    
def computeCost(A):
    counter = 0
    A = A.tolist()
    for i in range(len(A)):
        if A[i] != 0:
            counter+=1
    ratio = counter / len(A)
    ratio = ratio*100
    return ratio
    

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
    print("X shape: ", trainX.shape)
    print("Y shape: ", trainY.shape)
    # initialize weights and bias
    weight = np.ones(trainX.shape[1])
    bias = np.zeros(1)
    cost = []
    #epsilonCost = []
    # we will vectorize this, we dont want to see nested for loops
    for i in range(numitr):
        # in pyton, by default vector is a row vector
        Z = np.matmul(weight, trainX.T) + bias
        #print("Z's shape", Z.shape)
        # now we have 1xm vector, m is the number of training examples
        # apply sign function 
        output = signFunc(Z)
        #print("output", output.shape)
        
        # we only want to rotate if we missclassified
        #A = trainY - output
        A = trainY - output
        #print("A's shape", A.shape)
        costPer = computeCost(A)
        cost.append(costPer)
        #epsilonCost.append(costPer)
        if i % 100 == 0:
            cost.append(costPer)
            print("@ Epoch " + str(i/100) + " cost is: " + str(cost[i/100]))
        # now rotate towards missclassified point in the feature direction
        dW = (np.matmul(A.T, trainX))
        #print(dW.shape)
        dB = np.sum(A)
        # now update weight
        weight = weight + alpha*dW
        bias += alpha*dB
       
        #if abs(epsilonCost[i] - epsilonCost[i-1]) <= epsilon:
        #    return weight, bias, cost
    return weight, bias, cost
        

    

def main():
    X = preprocess('face_emotion_data_X.csv')
    Y = preprocess('face_emotion_data_y.csv')
    trainX, testX, trainY, testY = trainTestsplit(0.7, X, Y)
    
    weight, b, cost = trainWeights(trainX, trainY, 1000, 0.01, 0.1)
    
    
main()