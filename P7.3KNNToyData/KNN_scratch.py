# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 19:38:11 2019

@author: miru
"""

# import libraries
from math import sqrt
import pandas as pd
import numpy as np



def euclidean_distance(vector1, vector2):
    # takes in two vectors and computes their distance
    distance = 0.0
    # the last entry of the vector is {0,1} the classifier type so we dont want to add that
    #vector1 = np.array(vector1)
    
    #print(vector1)
    #vector2 = np.array(vector2)
    #print(type(vector1), type(vector2))
    for i in range(len(vector1) - 1):
        distance += (vector1[i] - vector2[i])**2
    return sqrt(distance)

def get_nbhd(trainset, test_vector, k):
    # k is the number of neighbours in the nbhd
    # initialize a list 
    dist = list()   # list() function will return an empty list
    for vector in trainset:
    #for i in range(len(trainset.iloc[:])):
        # append 2_norm distance between the two vectors
        #distance = euclidean_distance(vector, test_vector)
        distance = euclidean_distance(vector, test_vector)
        # now pair it with respective vector in the trainset
        dist.append( (vector, distance) ) 
    
    # dist is a list of tuples
    # using lambda expression we can sort the distlist coordinate wise
    dist.sort(key = lambda tup : tup[1])
    
    # create an empty list that will hold first k nbhds
    nbhd = list()
    for i in range(k):
        # we only want to grab the 1st coordinate of each tuple since that is the vector
        # dist = [(,) , (,) , .... , (,)]
        # dist[i] will grab the ith tuple
        nbhd.append(dist[i][0])
    return nbhd
        
def predict(trainset, test_vector, k):
    # grab the nbhd
    knn = get_nbhd(trainset, test_vector, k)
    # create a list to hold majority indicators
    majority = list()
    # get the majority result
    for nbh in knn:
        # the very last element holds the 
        majority.append(nbh[-1])
    # now we have the majority and we want to return the mode of the list
    mode = most_frequent(majority)
    return mode

def most_frequent(classes):
    # we need n-many cntrs for each type of class 
    cntr0 = 0
    cntr1= 0
    for i in classes:
        if (i == 0):
            cntr0 += 1
        else:
            cntr1 += 1
    if (cntr0 > cntr1):
        return 0
    else:
        return 1
    
# now we want to repeat above with a dataframe of testset
        
def KNN(trainset, testset, k):
    # initialize empty list
    result = list()
    
    #for i in range(len(testset.iloc[:])):
    for testvec in testset:
        classtype = predict(trainset, testvec, k)
        result.append( (test, classtype) )
    return result

###############################################################################
    

# below function is for performance analysis

def measure_accuracy(testset, trainset, k):
    # result is a list of tuples
    # we only want to last entry of each row of testset.
    # we will use the following metric: # correct points / # test points
    compareTo = KNN(trainset, testset, k)
    # init cntr to keep track of #correct points
    correct = 0
    actual = list()
    for testvec in testset:
        actual.append(testvec[-1])
    
    #for i in range(len(testset.iloc[:])):
    #for testvec in testset:
    for i in range(len(compareTo)):
        if (actual[i] == compareTo[i][1]):
            correct += 1
    return correct / float((np.size(testset, 0)))

###############################################################################

# below function is for data cleaning
    
def scale(dataframe):
    # TODO
    # We have to drop the last column of the dataframe because thats the classtype
    clean_df = (dataframe - np.mean(dataframe)) / np.std(dataframe)
    return clean_df

###############################################################################
    
# below function is for choosing appropriate k

import matplotlib.pyplot as plt

def choose_k(trainset, testset, itr):
    # initialize list to store accuracy
    accuracy = list()
    for k in range(1,itr):
        correct = measure_accuracy(testset, trainset, k)
        accuracy.append(correct)
    # plot using matplotlib
    plt.plot(range(1,itr), accuracy)
    plt.xlabel('number of neighbours')
    plt.ylabel('accuracy')
    plt.show

df = pd.read_csv('Classified Data.csv')

from sklearn.model_selection import train_test_split

#print(len(df.iloc[0:700]))
df.drop('Unnamed: 0', axis = 1, inplace = True)
df_train = df.iloc[0:700, :]
df_test = df.iloc[700:1000, :]
train = np.array(df_train)
test = np.array(df_test)
choose_k(train,test, 40)

# try to vectorize things YES IT WORKED

# the most stable range seems to be K = 20,...,25
# now we want to split the test set in to cross-validation set

def cross_validation(train_set, threshold):
    k = threshold
    end = np.size(train_set, 0)
    cross_validation = train_set[end - k: end]
    return cross_validation

validation_set = cross_validation(train, 200)

# resampling method

# see if we get a different optimal k if we train this on a different
# validation set.


    
    
    

    