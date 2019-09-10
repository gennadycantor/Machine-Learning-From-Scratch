# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:26:17 2019

@author: miru
"""

import random
import math
import numpy as np
#import seaborn as sns # data visualization we actually don't need this
import matplotlib.pyplot as plt
import pandas as pd # data prepocessing

def get_data(filename):
    fullData = pd.read_csv(filename)
    # get info
    #info = fullData.head()
    # get summary
    #summary = fullData.info()
    return fullData

def clean_data(data):
    # drop instances whose columns has atleast one missing value
    data = data[(data[['budget','genres', 'runtime', 'vote_average', 'vote_count', 'revenue']] != 0).all(axis=1)]
    data = data[(data[['budget','genres', 'runtime', 'vote_average', 'vote_count', 'revenue']] != '[]').all(axis=1)]    
    # we only need the following columns
    #budget, genres, runtime, vote_average, vote_count as features
    keepThese = ['budget', 'genres', 'runtime', 'vote_average', 'vote_count']
    revenue = 'revenue'
    label = data[revenue]
    #print(type(label))
    data = data[keepThese]
    for i in range(len(label)):
        if label.iloc[i] <= 50000000:
            label.iloc[i] = 0
        else:
            label.iloc[i] = 1

    return data, label

"split and group features: budget, runtime, vote average, vote count"
def uniform_split(data):
    # number of categories
    list_K = np.zeros(5)
    for i in range(len(list_K)):
        list_K[i] = random.randint(2,10)
    # minimum
    a = 0
    # maximum
    b = 0
    # smoothing
    epsilon = 0.01
    i = 0
    for column in data:
        a = data[column].min()
        b = data[column].max()
        K = list_K[i]
        for j in range(len(data[column])):
            if float((b + epsilon - a) / K) == 0:
                print(data[column])
            data[column].iloc[j] = math.floor( (data[column].iloc[j] - a) / float((b + epsilon - a)/K) ) 
        i += 1
    return data
            
def clean_genre(data):
    for column in data:
        if column == 'genres':
            for j in range(len(data[column])):
                # store string
                string = ((data['genres'].iloc[j]))
                # filter portion of string which is integer
                filteredStringToInt = float(filter(str.isdigit, string)[:2])
                # slice the first two integers (we will use this as subsets)
                # convert to string
                genreNum = filteredStringToInt
                data['genres'].iloc[j] = genreNum
    return data        

"""" entroy for the whole training set """    
def get_entropy(pv):
    # compute just one term
    # if we have more than 1 label or have different types of labels
    if pv != 0:
        H =  -1*( pv*np.log2(pv) )
    else:
        H = 0
    return H

""" compute infoGain(X, feature) per feature in {genre,....,vote_count} """
""" please note that this is a general version of calculating infogain 
    over all possible feature values. Of course, this complexity is not needed
    if we only have two labels """
def get_infoGain(X, y, feature):
    # initialize gain
    gain = 0
    numInstances = len(X)
    # per feature we need to store how many values it takes 
    featureVals = []
    for instance in X:
        if instance[feature] not in featureVals:
            featureVals.append(instance[feature])
    
    
    # per feature, we need to compute entropy(F_vi) for ith feature value
    entropy = np.zeros(len(featureVals)) # later we sum this up ofc
    # keep track of which feature value we are computing w.r.t entropy
    valueIndex = 0
    # find where each featurevalue is found in the instance[feature]
    # and its corresponding label = {0,1}
    # i.e. which sample has which feature value at Ith feature
    
    # we need this to compute how many times instances of some feature
    # with a particular feature value occur in the training set
    featureCounts = np.zeros(len(featureVals))
    # we need to return this... for output
    numKsplits = len(featureCounts)
    # we need this to see how many positive and negative splits we have per feature attribute
    posnegSplit = []
    # for each value in feature values for this particular feature
    for value in featureVals:
        # initialize dataIndex for each label that we will read from this
        # particulr feature value
        dataIndex = 0
        # init new labels list
        newLabels = []
        # for each instance in data set
        for instance in X:
            # if feature value is one of the feautre vallues in the prev.list
            if instance[feature] == value:
                # record how many times we see this particular feature value occur
                featureCounts[valueIndex] += 1
                # and store the label associated with this particular feature value of some feature
                # over all training instances
                newLabels.append(y[dataIndex])
            dataIndex += 1
        
        #initiazlze labelValues where we will use this to see how many of these values
        #occur over a training sample with a specific feature value
        labelValues = []
        # at this point we have a collection of labels whose value is either 0 or 1
        # and is associated with each instance with a specific feature value
        # so, for each label we hvae in the collection of labels
        for eachLabel in newLabels:
            # only store unique values
            if eachLabel not in labelValues:
                labelValues.append(eachLabel)
        # now initialize a list that will store the number of occurences of
        # each label across all samples with specific feature value
        labelCounts = np.zeros(len(labelValues)) # clearly only going to have
                                                        # same number of elements as types of vals
        labelIndex = 0
        # for every possible value {0,1} in our case
        for perLabel in labelValues:
            # loop through collection of labels (frequency) we stored previously
            for eachLabel in newLabels:
                # if a label in the frequency list matches specific type in labelValues
                if eachLabel == perLabel:
                    # incrememnt label count for that label
                    labelCounts[labelIndex] += 1
            labelIndex += 1
        
        # now compute entropy of samples over specific feature value for each feature
        for labelIndex in range(len(labelValues)):
            # probability value of true is just cardinality of occurence of label 0 divided by total 
            # and vice versa for false
            pv = labelCounts[labelIndex] / float(np.sum(labelCounts))
            posnegSplit.append((int(labelCounts[labelIndex])))
            entropy[valueIndex] += get_entropy(pv) 
            # now we have computed entropy(F_f1)
        # now compute summ of this over all feature values
        # we want to sum over all featureCounts over its valueIndex
        # recall featureCounts's each index records how many instances with specific feature
        # value occur under specific feature
        #if featureCounts[valueIndex] != 0: ### CAUTION
        gain += ( featureCounts[valueIndex] / numInstances )*entropy[valueIndex]
        valueIndex += 1

        
    return gain, numKsplits, posnegSplit
    
def makeTreelol(data,y):
    infoFile = open("output.txt", "w+")

    numInstances = len(data)
    numFeatures = len(data[0])
    gain = np.zeros(numFeatures)

    # first compute total entropy
    newClasses = []
    y = y.tolist()
    for eachClass in y:
        if newClasses.count(eachClass) == 0:
            newClasses.append(eachClass)

    labelFrequency = np.zeros(len(newClasses))
    totalEntropy = 0
    index = 0
    for eachClass in newClasses:
        labelFrequency[index] = y.count(eachClass)
        totalEntropy += get_entropy(labelFrequency[index] / float(numInstances))
        index += 1
    # initialize default label
    default = y[np.argmax(labelFrequency)]
    
    features = ['budget', 'genres', 'runtime', 'vote_average', 'vote_count']
    gain = np.zeros(numFeatures)
    for feature in range(numFeatures):
        g, k, posnegSplit = get_infoGain(data, y, feature)
        gain[feature] = totalEntropy - g
        infoFile.write(str(k) + ", "  + str((gain[feature])) + ", " + str(posnegSplit)  +  "\n" )
    argMaxFeat = np.argmax(gain)


    
def main():
    data = get_data('tmdb_5000_movies.csv')
    data, label = clean_data(data)
    # at this point, label is already cleaned
    data = clean_genre(data)
    # at this point, genre column is cleaned
    data = uniform_split(data)
    #print(data.iloc[0:5,:])
    # we uniformly split continuous feature variabels into K = 5 subsets
    # total instances are 3227 after removing missing values
    # use 70:30 split 2256:rest
    X = np.array(data)
    X = X.astype(np.float)
    # convert to float
    Y = np.array(label)
    Y = Y.astype(np.float)
    print("plase: ", len(X[:,1]))
    #X_train = X[:2200,:]
    #X_test = X[2200:,:]
    #Y_train = Y[:2200]
    #Y_test = Y[2200:]
    # feature order [budget genres runtime vote_avg vote_count]
    #################### preprocessing is done. ####################
    makeTreelol(X, Y)
    
main()
