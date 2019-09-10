# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:14:10 2019

@author: miru
"""

# gradient descent algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''

until convergence {
        temp = c - a*1/2n*[ sum(h(x) - y )^2]
        c = temp
}

what we need training set data, X and Y
we need the learning rate: a
n is the number of features
'''


df = pd.read_csv('USA_Housing.csv')

# becareful for indexing; upper bound is excluded so  really we are grabbing from 0 to 4
X = df.iloc[:,0:5]
y = df.iloc[:,5]
#X.insert(0, column = '1s', value = 1)
X = np.array(X)
# we will want to feauture scale
X = X[:,1:5]

X = (X - np.mean(X)) / np.std(X)
#Y = np.array(y)
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis = 1)
theta = np.zeros(X.shape[1])



X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.4, random_state = 101 )

from numpy import linalg as la

def compute_cost(X,Y,theta):
    sumof = np.dot(X,theta) - Y
    cost = la.norm(sumof)
    return cost

def grad_descent(X,Y,a,theta,itr):
    '''
    this is a vectorized implementation with choice of the following params
    a = 0.01, number of iterations = 1000
    '''
    plotcost = np.zeros(itr)
    m = len(Y)
    for i in range(itr):
        #temp = np.dot(X,theta) - Y
        temp = np.matmul(X,theta) - Y
        if i == 1:
            print("d", temp.shape)
            print("y", Y.shape)
        #temp = np.dot(X.T,temp)
        temp = np.matmul(temp, X)
        theta = theta - (a/m)*temp
        plotcost[i] = compute_cost(X,Y,theta)
    return theta,plotcost

coeff,estimates = grad_descent(X_train,y_train,0.01,theta,1000)
   
plt.plot(estimates)
plt.xlabel('num iteratinos')
plt.ylabel('cost')
plt.show
        




            


    
           
               
        
                  








            
    