# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:35:39 2019

@author: miru
"""

import numpy as np
from csv import reader
from math import exp
import matplotlib.pyplot as plt


""" read csv """
def get_data(filename):
    data = list()
    #open file for reading
    with open(filename, 'r') as file:
        read_csv = reader(file)
        #for each row in csv
        for row in read_csv:
            #append it to our "list"
            if row[0] == '4' or row[0] == '6':
                data.append(row)
    return data

def get_final(filename):
    data = list()
    #open file for reading
    with open(filename, 'r') as file:
        read_csv = reader(file)
        #for each row in csv
        for row in read_csv:
            #append it to our "list"
            data.append(row)
    return data
            
    

""" preprocess and clean data """
def clean_data(data):
    # first turn our list into a matirx
    X = np.array(data)
    # change to float 
    X = X.astype(np.float)
    # now change the label
    #X = X[:,1:] / float(255)
    for x in range(0, X.shape[0]):        
        if X[x,0] == 4.0:
            #counter4 += 1
            X[x,0] = 0
        elif X[x,0] == 6.0:
            #counter6  += 1
            X[x,0] = 1
   
    return X             

""" compute cost """
def compute_cost(a,Y):
    cost = np.zeros(len(a))
    for i in range(len(a)):
        if Y[i] == 0:
            if (1-a[i]) < 0.0001:
                cost[i] = 1*100
            else:
                cost[i] = -1*np.log(1-a[i])
        elif Y[i] == 1:
            if (a[i]) < 0.0001:
                cost[i] = 1*100
            else:
                cost[i] = -1*np.log(a[i])
    cost = 1*np.sum(cost)
    return cost

""" apply sigmoid/activation function to inner product: w.Tx """
def activation(theta):
    theta = 1 / (1 + np.exp(-theta))
    return theta

""" gradient descend and minimize our cost function """
def gradient_descent(X,Y, alpha, numitr, coeff):

    b = np.zeros(1)
    # we are going to append cost/epoch in the list below
    cost = []
    for i in range(numitr):
        Z = np.matmul(coeff, X.T) + b
        a = activation(Z)
        costPer = compute_cost(a,Y)
        q = (a - Y)
       
        deltaW = np.matmul(q,X)
        deltaB = np.sum(q)
        coeff -= alpha*deltaW
        b -= alpha*deltaB
        # only append every 100th iteration/epoch
        if i % 100 == 0:
            cost.append(costPer)
            print("@ Epoch " + str(i/100) + " cost is: " + str(cost[i/100]))
    return coeff, b, cost

""" fit our model after training on train set and evaluate accuracy """
def evaluate(weights, bias, testX, testY):
    Z = np.matmul(weights, testX.T) + bias
    a = activation(Z)
    outputs = np.zeros(len(a))
    print("dimension of a:", a.shape)
    for i in range(len(a)):
        if a[i] > 0.5:
            a[i] = 1
            outputs[i] = 6
        else:
            a[i] = 0
            outputs[i] = 4
    # now evaluate
    counter = 0
    numInstances = len(testY)
    for i in range(len(a)):
        if a[i] == testY[i]:
            counter += 1
    score = counter / float(numInstances)
    return score, outputs

""" final eval """
def evaluate_final(weights, bias, X):
    Z = np.matmul(weights, X.T) + bias
    a = activation(Z)
    outputs = np.zeros(len(a))
    print("dimension of a:", a.shape)
    for i in range(len(a)):
        if a[i] > 0.5:
            a[i] = 1
            outputs[i] = 6
        else:
            a[i] = 0
            outputs[i] = 4
    return outputs
    
""" normalize our train and test set """
def normalize(X):
    #X = X[:, 1:] / float(255)
    X = X / float(255)
    return X
""" split our train and test into trainLabel/trainFeauture & testLabel/testFeature """
def partition(X):
    Feature = X[:, 1:]
    Label = X[:,0]
    return Feature,Label

""" main function """
def main():
    train = get_data('mnist_train.csv')
    test = get_data('mnist_test.csv')
    # clean data
    train = clean_data(train)
    test = clean_data(test)
    trainX, trainY = partition(train)
    testX, testY = partition(test)
    trainX = normalize(trainX)
    testX = normalize(testX)
    print("we should have this many weights: ", trainX.shape[1])
    
    # initialize coefficients to a 0 vector
    coeff = np.zeros(trainX.shape[1])
    
    ''' Hyperparameter Tuning Block'''
    #alphaList = np.zeros(2)
    #alphaList[0] = 0.1
    #alphaList[0] = 0.01
    #alphaList[1] = 0.001
    # we will use numitr = 1000
    # alpha = 0.01 seems good
    #alpha = 0.01
    alpha = 0.001
    #for elem in alphaList:
    #    weights,bias = gradient_descent(trainX, trainY, elem, 1000, coeff)
    #    print(weights, bias)
    # try numitr = 2000 with alpha = 0.01
    weights, bias, cost = gradient_descent(trainX, trainY, alpha, 1000, coeff)
    
    #write to file
    f = open("weights.txt", "w+")
    f.write(str(int(bias)))
    for i in range(len(weights)):
        f.write(str(int(weights[i])) + "\n")
    #f.write(str(bias))
    f.close()
    
    ''' dummy ''' 
    #write to file
    ff = open("weightshehe.txt", "w+")
    for i in range(len(weights)):
        ff.write(str(float(weights[i])) + "\n")
    ff.write(str(bias))
    ff.close()
    
    
    
    ''' display and plot cost '''
    plt.plot(cost)
    plt.xlabel(' number of iterations (100s) ')
    plt.ylabel('cost')
    plt.show
    
    ''' evaluate and print accuracy of our model '''
    accuracy, outputs = evaluate(weights, bias, testX, testY)
    accuracy = accuracy * 100
    print("accuracy of the model is: " + str(accuracy) + " %")
    
    #write to file
    fo = open("IntermediateOutput.txt", "w+")
    for i in range(len(outputs)):
        fo.write(str(int(outputs[i])) + "\n")
    fo.close()
    
    ''' final '''
    test4 = get_final('test_4.csv')
    test6 = get_final('test_6.csv')
    # convert each to matrix and concetenate
    X_4 = np.array(test4)
    X_6 = np.array(test6)
    
    final = np.concatenate((X_4, X_6), axis=0)
    # convert to float
    final = final.astype(np.float)
    # normalize
    #final = normalize(final)
    final = final / float(255)
   
    finalOutputs = evaluate_final(weights, bias, final)
    
    #counters
    counter4 = 0
    counter6 = 0
    ''' see number of 4's and 6's '''
    for i in range(len(finalOutputs)):
        if finalOutputs[i] == 4.0:
            counter4 += 1
        else:
            counter6 += 1
    print("number of 4's: ", counter4)
    print("number of 6's: ", counter6)
   
    
    foFinal = open("output.txt", "w+")
    for i in range(len(finalOutputs)):
        foFinal.write(str(int(finalOutputs[i])) + "\n")
    foFinal.close()
    
main()    