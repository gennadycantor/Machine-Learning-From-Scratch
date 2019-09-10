# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:53:10 2019

@author: miru
"""
import random
import numpy as np
from random import randrange

# each flapis bird moving
def genBirdActions(birdActions):
    #a_it = 1 with pv = 1/3 
    #a_it = 0 with pv = 2/3
    
    # we have 100 birds
    for i in range(100):
        # each moves 100 times
        for j in range(100):
            # create a list of random numbers 1,2,3
            randomInt = randrange(1,4)
            if randomInt == 1:
                action = 1
            else:
                action = 0
            birdActions[i,j] = action
    
    return birdActions

def mutate(newGenKiwis):
    
    mutation = np.zeros((100,100))
    m = 0.01
    for i in range(100):
        for j in range(100):
            randomNum = random.random()
            # this is like almost unlikely 
            if randomNum < m:
                # switch back and forth 0 and 1
                if newGenKiwis[i,j] == 0:
                    mutation[i,j] = 1
                else:
                    mutation[i,j] = 0
            else:
                mutation[i,j] = newGenKiwis[i,j]
    
    return mutation            

def getParent(uni, cdf):
    for m in range(len(cdf)):
        if uni <= cdf[m]:
            return m
    return len(cdf) - 1

def getCumul(repPv):
    cdf = np.zeros(len(repPv))
    counter = 0
    for i in range(len(repPv)):
        counter += repPv[i]
        cdf[i] = counter
    return cdf


# each column corresponding to that birds action of some row's game
def geneticAlg(someGame, birdActions, x, y):
    counter = 0
    kiwiScore = np.zeros(100)
    
    # initizlize score. we need to keep track of these
    curScore = -1
    prevScore = 0
    
    # we need to perform the algorithm until s_i, the average score does not change
    while (prevScore != curScore):
        # ok this is way too slow
        if counter > 500:
            break
        counter += 1
        print("1", counter)
        kiwiScore = np.zeros(100)
        # get z_it
        #
        #
        # we need to get score for each kiwi bird (go NZ!)
        for i in range(100):
            print("2", counter)
            #init gamescore
            gameScore = 0
            # initial position is 5
            currPos = 5
            # for each action bird makes (100 "actions")
            for j in range(100):
                
                if birdActions[i,j] == 0:
                    currPos -= y
                if birdActions[i,j] == 1:
                    currPos += x
                # now check if there is no pipe and bird is in the correct position
                if someGame[j] == 0:
                    #position should be between [0,10]
                    if currPos > 0 and currPos <10:
                        gameScore +=1
                # now if there is a pipe and bird does not hit the pipe
                elif  someGame[j] != 0:
                    if currPos > someGame[j] and currPos < someGame[j] + 2:
                        gameScore += 1
                else:
                    # bird died
                    break
            kiwiScore[i] = gameScore
        
        totalScore = np.sum(kiwiScore)
        # switch scores
        prevScore = curScore
        curScore = totalScore / float(100)
        
        # lets now compute Reproduction pv
        repPv = np.zeros(100)
        
        for k in range(100):
            print("3", counter)
            repPv[k] = kiwiScore[k] / float(totalScore)
        
        # get CDF
        cdf = getCumul(repPv)
        
        
        # do genbird but for new gen
        newGenKiwis = np.zeros((100,100))
        
        for l in range(50):
            print("4", counter)
            # get 2 float numbers
            uni_1 = random.random()
            uni_2 = random.random()
            
            # compute parent
            
            parent1 = getParent(uni_1, cdf)
            parent2 = getParent(uni_2, cdf)
            
            if parent1 == parent2:
                print("hehehe")
                print("zoomzoom")
                while parent1 == parent2:
                    print("7", counter)
                    #uni_1 = np.random.uniform(0,100)
                    uni_1 = random.random()
                    print("heeeeee")
                   # parent1 = getParent(uni_1, cdf)
                    parent1 = getParent(uni_1, cdf)
            
            parent1Game = birdActions[parent1]
            parent2Game = birdActions[parent2]
            
            # store death index
            deathIndex = kiwiScore[parent1] 
            # compute cross over
            getChildren = np.zeros((2,len(parent1Game)))
            print("youuu")
            for p in range(len(parent1Game)):
                print("8", counter)
                if (p < deathIndex):
                    getChildren[0, p] = parent1Game[p]
                    getChildren[1,p] = parent2Game[p]
                else:
                    getChildren[0,p] = parent2Game[p]
                    getChildren[1,p] = parent1Game[p]
            
            
            newGenKiwis[2*l] = getChildren[0]
            newGenKiwis[(2*l) + 1] = getChildren[1]
        print("111111")
        # now mutate
        newGenKiwis = mutate(newGenKiwis)
        #store newGenKiwis to previous gen
        birdActions = newGenKiwis
    
    return birdActions, kiwiScore
        


def main():
    # game sequences
    game1 = '0000000050000000000000000000060000000000000000000000000760000060000600000000402000000030060070707000'
    game2 = '0000000000000000607070000500000000007000000000000000007700000000002200000005040000000400200000011201'
    game3 = '0000000000000000000400005000000000000000050500054500060777007000000000000200000100011000030000000020'
    game4 = '0000000000000000000000000000000000400000000400004000000000000400400050400000000005500500670070070550'
    game5 = '0000000000000000000000000200021000000100002000000000020000000000070004000500000000000400000000230003'
    x = 0.5
    y = 0.25
    row1 = np.zeros(len(game1))
    row2 = np.zeros(len(game1))
    row3 = np.zeros(len(game1))
    row4 = np.zeros(len(game1))
    row5 = np.zeros(len(game1))
    
    for i in range(len(game1)):
        row1[i] = int(game1[i])
    for i in range(len(game1)):
        row2[i] = int(game2[i])
    for i in range(len(game1)):
        row3[i] = int(game3[i])  
    for i in range(len(game1)):
        row4[i] = int(game4[i])
    for i in range(len(game1)):
        row5[i] = int(game5[i])
    
    games = []
    games.append(row1)
    games.append(row2)
    games.append(row3)
    games.append(row4)
    games.append(row5)    
   
    # init action matrix
    birdActions = np.zeros((100,100))
    birdActions = genBirdActions(birdActions)
    
    scoreList = []
    
    for j in range(5):
        genActions, score = geneticAlg(games[j], birdActions,x,y)
        if j == 1:
            print(genActions)
            #break
            foFinal = open("populationZ.txt", "w+")
            for i in range(len(game2)):
                foFinal.write(str(int(games[1][i])))
            foFinal.write("\n")
            for a in range(100):
                for b in range(100):
                    foFinal.write(str(int(genActions[a,b])))
                foFinal.write("\n")
            foFinal.close()
        genActions = genBirdActions(genActions)
        scoreList.append(score)
    print(len(scoreList))
    folol = open("outputZ.txt", "w+")
    for l in range(5):
        for m in range(100):
            folol.write(str(int(scoreList[l][m])) + ",")
        folol.write("\n")
    folol.close()
    
main()
    
    
    
    