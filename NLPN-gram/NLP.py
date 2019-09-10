# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:44:11 2019

@author: miru
"""
import random
import numpy as np
import pandas as pd
import re
import collections
import string

""" reads a txt file and returns as an array """
def preprocess1(filename):
    txt = open(filename, 'r')
    txtArray = txt.readlines() # reads by line and stores in to list
    return txtArray

""" reads a string and does appropriate removal """
def preprocess2(fromList):
    text = listToString(fromList)
    # now remove stop words
    text = re.sub(' +', ' ', text)
    text = re.sub('[^a-z]', ' ', text)
    
    
    return text

def makedict():
    numlist=[]
    for i in range(27):
        numlist.append(i)
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    zipDict = zip(numlist, someList)
    return dict(zipDict)

def computeTransitionPV(transMatrix, probDict, unigram):
    orderedDict = collections.OrderedDict(sorted(probDict.items()))
    for i in range(len(transMatrix.index)):
        for j in range(len(transMatrix.columns)):
            if orderedDict.has_key(transMatrix.index.values[i] + transMatrix.columns.values[j]):
                #print(1)
                transMatrix.iloc[i,j] = orderedDict[transMatrix.index.values[i] + transMatrix.columns.values[j]]
            else:
                transMatrix.iloc[i,j] = 1 / float(unigram[transMatrix.index.values[i]] + 27)
    return transMatrix

def computeTransitionPV2(transMatrix, probDict, bigram):
    orderedDict = collections.OrderedDict(sorted(probDict.items()))
    orderedDictBi = collections.OrderedDict(sorted(bigram.items()))
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    for i in range(len(transMatrix.index)):
        for j in range(len(transMatrix.columns)):
            for k in range(len(someList)):
                if orderedDict.has_key(transMatrix.index.values[i] + transMatrix.columns.values[j] + someList[k]):
                    #print(1)
                    transMatrix.iloc[i,j] = orderedDict[transMatrix.index.values[i] + transMatrix.columns.values[j] + someList[k]]
                else:
                    if (orderedDictBi.has_key(transMatrix.index.values[i] + transMatrix.columns.values[j])):
                        transMatrix.iloc[i,j] = 1 / float(bigram[transMatrix.index.values[i] + transMatrix.columns.values[j]] + 27)
                    else:
                        transMatrix.iloc[i,j] = 1 / float(0 + 27)
                        
    return transMatrix

def transMatrix2(probDict, unigram):
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    #print(someList)
    rowDict = makedict()
    transMatrix = pd.DataFrame(np.zeros((27,27)), columns = someList)
    transMatrix.rename(index = rowDict, inplace=True)
    transMatrix = computeTransitionPV(transMatrix, probDict, unigram)
    print(transMatrix.head())
    return transMatrix

def transMatrix3(probDict, biigram):
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    #print(someList)
    rowDict = makedict()
    transMatrix = pd.DataFrame(np.zeros((27,27)), columns = someList)
    transMatrix.rename(index = rowDict, inplace=True)
    transMatrix = computeTransitionPV2(transMatrix, probDict, biigram)
    
    return transMatrix

def getProbLP1(freqDict, M):
    probDict = {token : (freqDict[token] + 1) / float((M + 27)) for token in freqDict}
    return probDict
    
def getProbLP2(freqDict, unigramDict):
    probDict = {token : (freqDict[token] + 1) / float((unigramDict[token[0]] + 27)) for token in freqDict}
    return probDict

def getProbLP3(freqDict, bigramDict):
    probDict = {token : (freqDict[token] + 1) / float((bigramDict[token[0] + token[1]] + 27)) for token in freqDict}
    return probDict

def recordOccurence(tokenList):
  
    # Creating an empty dictionary  
    freq = {elem : 0  for elem in tokenList}
    for item in tokenList: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    return freq

def ngramTable(fstr, n):
    getTokens = list(fstr)
    # zip it into unique consecutive concats    
    ngramsDict = zip(*[getTokens[i:] for i in range(n)])
    return ["".join(gram) for gram in ngramsDict]
    #return ngramsDict
    
def uniGram(fstr):
    tokenTable = ngramTable(fstr, 1)
    freqDict = recordOccurence(tokenTable)
    M = 0
    for key in freqDict:
        sup = freqDict[key]
        M += sup
    probDict =  getProbLP1(freqDict, M)
    
    return M, freqDict, probDict

def biGram(fstr, uniGram):
    tokenTable = ngramTable(fstr, 2)
    #print(type(tokenTable))
    # turn it into a dictionary whos key is the number of occurences
    freqDict = recordOccurence(tokenTable)
    #print(freqDict)
    probDict = getProbLP2(freqDict, uniGram)
    #print("2", type(freqDict))
    orderedDict = collections.OrderedDict(sorted(probDict.items()))
    #orderedDict = dict(orderedDict)
    #print(orderedDict)
    #print(type(orderedDict))
    #print(len(orderedDict))
    return freqDict, probDict
    #transMatrix(tokenTable)

def triGram(fstr, bigramDict):
    tokenTable = ngramTable(fstr, 3)
    freqDict = recordOccurence(tokenTable)
    probDict = getProbLP3(freqDict, bigramDict)
    orderedDict = collections.OrderedDict(sorted(probDict.items()))
    #print("3", type(freqDict))
    return freqDict, probDict
    #print(tokenTable)
    
def listToString(someList):
    s = ""
    for i in range(len(someList)):
        s += someList[i]
    return s

def getCDF(alphabet, transMatrix):
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    #transMatrix = np.array(transMatrix)
    #print("hehe", someList.index(alphabet))
    #index = int(someList.index(alphabet))
   # print(transMatrix[index])
    cdf = np.zeros(27)
    cumulPV = 0
    #cumullist = transMatrix[index]
    #print(type(transMatrix))
    for i in range(27):
        cumulPV = cumulPV + transMatrix.loc[alphabet][i]
        cdf[i] = cumulPV
    return cdf

def getArgMax(alphabet, cdf, flag ,transMatrix2, transMatrix3):
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    for i in range(len(someList) - 1):
        # find the appropriate cdf range
        if alphabet <= cdf[i]:
            if flag == 1:
                #print(type(transMatrix2.columns.values[i]))
                #return  transMatrix2.columns.values[i]
                return someList[i]
            else:
                #print(type(transMatrix2.columns.values[i]))
                #return  transMatrix3.columns.values[i]
                return someList[i]
                
def generateSentence(bigram, trigramDict, transMatrix2, transMatrix3):
    # we are generating sentences that starts with a,b,c,d....,z per alhabet
    # and persentence we will have characters
    sentenceList = []
    someList = list('abcdefghijklmnopqrstuvwxyz ')
    # 26 sentences
    # we begin a sentence from each of these alphabets
    for i in range(len(someList)-1):
        
        # we need to do this 10 times for each character
        for j in range(10):
            s = someList[i]    
            #print("damn", s)
            # generate random variable u
        
            # legnth of 100
            while len(s) < 100:
                
                    # generate random 2nd letter 
                randomCh = np.random.uniform(0,1,1)
                #print(len(randomCh))
                #print(randomCh)
                randomCh = float((randomCh))
                # now check if the length of the string is 1 or more
                #init cdf per row
                #cdf = np.zeros(27)
                if len(s) == 1:
                    # then use bigram MLE
                    cdf = getCDF(s[0], transMatrix2)
                    flag = 1
                # if we have more than 1 letter in the sentecne now
                # use trigram MLE
                else:
                    cdf = getCDF(s[len(s)-1],transMatrix3)
                    flag = 2
                # now get argmax
                argmaxChar = getArgMax(randomCh, cdf, flag, transMatrix2, transMatrix3)
                if (type(argmaxChar) != str):
                    argmaxChar = ' '
                    # print("damn" + str(i) + "ehe" + str(j))
                s = s + argmaxChar
            sentenceList.append(s)
    return sentenceList
                
            
            
    
    
def main():
    # read the text and convert to an array
    txtArr = preprocess1('memento.txt')
    someStr = preprocess2(txtArr)
    M, unigramDict, probDict = uniGram(someStr)
    bigramDict, probDict2 = biGram(someStr, unigramDict)
    trigramDict, probDict3 = triGram(someStr, bigramDict)
    # probdict for unigram
    #print(type(unigramDict))
    #print(type(bigramDict))
    print(type(trigramDict))
    probDict1 = getProbLP1(unigramDict, M)
    probDict2 = getProbLP2(bigramDict,unigramDict)
    probDict3 = getProbLP3(trigramDict, bigramDict)
    #probDict3 = getProblP3(trigramDict, bigramDict)
    # now probDict 2 is the proability dictionary
    transitionMatrix2 = transMatrix2(probDict2, unigramDict)
    print(transitionMatrix2.info())
    #print((transitionMatrix2.iloc[1,3]))
    transitionMatrix2.to_csv('bigram.txt', header = None, index = False, sep=',', float_format = '%.8f')
    transitionMatrix3 = transMatrix3(probDict3, bigramDict)
    transitionMatrix3.to_csv('trigram.txt', header = None, index = False, sep=',', float_format = '%.8f')
    sentence = generateSentence(bigramDict, trigramDict, transitionMatrix2, transitionMatrix3)
    #print(sentence[0])
    print(len(sentence))
    foFinal = open("output.txt", "w+")
    for i in range(len(sentence)):
        foFinal.write(str((sentence[i])) + "\n")
    foFinal.close()
    
    return 0
    

main()
    