# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:54:05 2019

@author: miru
"""
from random import seed
from random import randrange
from csv import reader


# load the csv 
def load_csv(filename): 
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
          
# change to float      
def toFloat(dataset, column):
    
    for row in dataset:
        
        row[column] = float(row[column].strip())
        
  
def cvSplit(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# get accuracy
def getAccuracy(actual, prediction):
    counter = 0
    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            counter +=1
    return counter / float(len(actual)) * float(100)

# evaluate our decision tree using CV
def evaluate(dataset, alg, num_folds, *args):
    folds = cvSplit(dataset, num_folds)
    score = list()
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        train = sum(train, [])
        test = list()
        for x in fold:
            copyx = list(x)
            test.append(copyx)
            copyx[-1] = None
        prediction = alg(train, test, *args)
        actual = [x[-1] for x in fold]
        accuracy = getAccuracy(actual, prediction)
        score.append(accuracy)
    return score

# this is what we did for decision stump
def splitting(index, value, dataset):
    l = list()
    r = list()
    for row in dataset:
        if row[index] < value:
            l.append(row)
        else:
            r.append(row)
    return l,r

# alternative to computing info gain: giniIndex
# compute giniIndex

def gini_index(groups, classes):
    # get all instances when we split
    n_instances = float(sum([len(group) for group in groups]))
    # sum the index this is very analgous to how we computed infogain
    # for decision stump
    gini = float(0)
    for group in groups:
        s = float(len(group))
        # do not devide by 0
        if s == 0:
            continue
        score = float(0)
        # compute score w.r.t each class
        for classVal in classes:
            prob = [row[-1] for row in group].count(classVal) / float(s)
            score += prob*prob
        # divide by cardiniality
        gini += (float(1) - score) * (s / n_instances)
    return gini

# get the Best split, i.e. argmax infogain

def getSplit(dataset):
    classVal = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 9999, 9999, 9999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = splitting(index, row[index], dataset)
            gini = gini_index(groups, classVal)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    # zip it to dict
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# make leaf nodes
def leafNodes(group):
    results = [row[-1] for row in group]
    return max(set(results), key = results.count)

###### HIGH LIGHT: BUILD ACTUAL TREE, NOT STUMP #######
    
# really analogous to bs tree #    
def split(node, max_depth, min_size, depth):
    l,r = node['groups']
    del(node['groups'])
    # check if we do not split
    if not l or not r:
        node['left'] = node['right'] = leafNodes(l + r)
        # and we are doen
        return
    # now check whether we reached maximal depth of tree
    if depth >= max_depth:
        node['left'], node['right'] = leafNodes(l), leafNodes(r)
        # and done
        return
    # recurse down for left and processs left child
    if len(l) <= min_size:
        node['left'] = leafNodes(l)
    else:
        node['left'] = getSplit(l)
        split(node['left'], max_depth, min_size, depth + 1)
    # do the same for right
    if len(r) <= min_size:
        node['right'] = leafNodes(r)
    else:
        node['right'] = getSplit(r)
        split(node['right'], max_depth, min_size, depth + 1)
        
# actually build the tree
    
def build(train, max_depth, min_size):
    root = getSplit(train)
    split(root, max_depth, min_size, 1)
    return root

# debugging purpose, we just print tree
def printTree(node, depth= 0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value']))) 
        printTree(node['left'], depth + 1)
        printTree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# well, we built a tree, so lets make a prediction
def predict(node, row):
    if row[node['index']] < node['value']:
        # recurse down until you reach leaves
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
def DTALG(train, test, max_depth, min_size):
    tree  = build(train, max_depth, min_size)
    printTree(tree)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)
    
def main():
    # i'm setting seed to debug, when you run it, feel free to remove it
    seed(1)
    dataset = load_csv('banknote.csv')
    # preprocess
    for i in range(len(dataset[0])):
        toFloat(dataset, i)
    
    # numfold for cross valid
    # set it to whatever you want! I'm setting mine to 6
    numFold = 6
    # max depth for tree
    # set it to whatever you want! I'm setting mine to 5
    max_depth = 5
    min_size = 9
    
    scores = evaluate(dataset, DTALG, numFold, max_depth, min_size)
    print("Scores: %s" %scores)
    print("average accuracy: %3f%%" % (sum(scores)/ float(len(scores))))
    
    
main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
