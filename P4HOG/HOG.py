# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:43:46 2019

@author: miru
"""
#import cv2
import os
from PIL import Image
import pandas as pd
import numpy as np
import math

""" convert to grayscale """
def convertToGray():
    #gsList = []
    for i in range(1,9):
        #pil_im = Image.open('image' + str(i) + '.jpg').convert('L')
        pil_im = Image.open('image' + str(i) + '.jpg').convert('L')
        pil_im.save('grayscale' + str(i) + '.jpg')   

""" convert to 2dArray """
def convertToArray():
    arrayList = []
    for i in range(1,9):
        #im = np.array(Image.open('image' + str(i) + '.jpg').convert('L'),'f')
        im = np.array(Image.open('grayscale' + str(i) + '.jpg'),'f')
        arrayList.append(im)
    
    for i in range(len(arrayList)):
        arrayList[i] = arrayList[i] * (1/float(255))
    return arrayList

# =============================================================================
# def imagePreProcess():
#     arrayList = []
#     newArrayList = []
#     for i in range(1,7):
#         img_file = 'image' + str(i) +'.jpg'
#         #img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
#         #img = cv2.imread(img_file, cv2.IMREAD_COLOR)
#         img = img.astype(np.float)
#         arrayList.append(img)
#     #print((arrayList[0].shape))
#     for i in range(len(arrayList)):
#         width = 150
#         height = 150
#         newArray = arrayList[i] / float(255)
#         newArrayList.append(newArray)
# =============================================================================
# =============================================================================
#         newArray = np.zeros((width, height))
#         for j in range(width):
#             for k in range(height):
#                 
#                 #newArray[j,k] = sum(arrayList[i][j,k]) / 3
#                 newArray = newArray / float(255)
#                 newArrayList.append(newArray)
# =============================================================================
   # return newArrayList
    

""" convolve two matrices: sobel and pic """
def convolve(I, sobel, middleX, middleY):
    # init taretIndex
    #print(I.shape)
    targetIndex = 0
    #print(sobel.shape[0])
    for i in range(sobel.shape[0]):
        #sweetSpotX = middleX + i - 1
        for j in range(sobel.shape[1]):
            sweetSpotX = middleX + i
            sweetSpotY = middleY + j
            # element wise between a 3x3 submatrix of I
            # and 3x3 sobel matrix
            # we need to make sure, we don't go out of the "padding" zone of
            # the image
            # only compute element prod when the overlap does not
            # escape I's dim, else its just 0
            # so long as the given Image's position + sobel's dim1  - 1
            # is non-negative and does not exceep end of dim1 of Image,
            if sweetSpotX >= 1 and sweetSpotX <= I.shape[0]:
                # and the same for dim2 of image range
                if sweetSpotY >= 1 and sweetSpotY <= I.shape[1]:
                    #print("hehe")
                    # compute the elementwise convolutoin
                    targetIndex += I[sweetSpotX - 1, sweetSpotY - 1]*sobel[i, j]
                    
    return targetIndex
    
""" get resulting gradient matrix """
def getGradient(picArray):
    # define sobel matrix
    # test with only one pic2darray
    W_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    W_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    grad_xI = np.zeros((picArray.shape[0], picArray.shape[1]))
    grad_yI = np.zeros((picArray.shape[0], picArray.shape[1]))
    # now compute convolution
    for i in range(picArray.shape[0]):
        for j in range(picArray.shape[1]):
            grad_xI[i,j] = convolve(picArray, W_x, i, j)
            grad_yI[i,j] = convolve(picArray, W_y, i, j)
            
    
    # now we are done constructing gradient matrix
    # as shown in the slides,
    finalGrad = np.sqrt(np.square(grad_xI) + np.square(grad_yI))
    return finalGrad
    
    
""" get resulting direction matrix """
def getMagnitude(picArray):
    # define sobel matrix
    # test with only one pic2darray
    W_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    W_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    grad_xI = np.zeros((picArray.shape[0], picArray.shape[1]))
    grad_yI = np.zeros((picArray.shape[0], picArray.shape[1]))
    # now compute convolution
    for i in range(picArray.shape[0]):
        for j in range(picArray.shape[1]):
            grad_xI[i,j] = convolve(picArray, W_x, i, j)
            grad_yI[i,j] = convolve(picArray, W_y, i, j)
    # now we are done constructing directinoal matrix
    # as shown in the slides
    finalDir = np.arctan2(grad_xI, (grad_yI))
    #finalDir = math.atan2(grad_xI, grad_yI)
    return finalDir

def partitionGrad(gradMatrix):
    pList = []
    colIndex = 15
    rowIndex = 15
    for i in range(rowIndex + 1):
        for j in range(rowIndex + 1):
            
            target = gradMatrix[15*i:15*(i+1), :]
            pList.append(target) 
            return pList
   
def partitionDir(dirMatrix):
    pList = []
    colIndex = 15
    rowIndex = 15
    for i in range(colIndex + 1):
        for j in range(rowIndex + 1):
            target = dirMatrix[15*i:15*(i+1), 15*j:15*(j+1)]
            pList.append(target)
     
    return pList
    
def partitionImage(imageMatrix):
    pList = []
    colIndex = 15
    rowIndex = 15
    for i in range(colIndex + 1):
        for j in range(rowIndex + 1):
            target = imageMatrix[15*i:15*(i+1), 15*j:15*(j+1)]
            pList.append(target)
                 
    return pList

def gradBin(submatrixI, submatrixG, submatrixD, binList):
    # init gradperbin
    # 15x15 many submatrices
    
    # divide into 225 submatrices
    for i in range(0, submatrixI.shape[0], 10):
        for j in range(0, submatrixI.shape[1], 10):
            #target = submatrixI[15*i:15*(i+1), :]
            # per subdivision of a matrix
            #k = counteri
            #l = counterj
            # init a array of size 8 for each bin category
            BinCategory = np.zeros(8)
            # for every row and column of each submatrix
            # sum up cells of finalGrad w.r.t similar directions
            for k in range(i, 10 + i):
                for l in range(j, 10 + j):
                    # we need to access each cell in directionmatrix
                    # and check is less than 0 or not
                    curr_dirCell = submatrixD[k,l]
                    if (curr_dirCell - math.pi) == 0:
                        curr_dirCell = 0
                    elif (curr_dirCell < 0):
                        curr_dirCell += math.pi
                    # then compute bin in [0,...,7]
                    rhs = math.floor(curr_dirCell / float(math.pi / float((8))) )
                    # and convert to int
                    Bin = int(rhs % 8)
                    # and add the sum of finalgradient with similar direction to bin category 
                    #print((submatrixG[k,l]))
                    BinCategory[Bin] += submatrixG[k,l]
                    #print(BinCategory[Bin])
            #j += 10
            # we have completed, at this point, 1 10x10 submatrix
            # so now append this to the actual BinList, which will contain 255*8 numbers
            for b in range(len(BinCategory)):
                #f.write(str(int(b)) + ',')
                binList.append(BinCategory[b])
        #i += 10
    return binList
    
    
def hogBin(picArray, finalGrad, finalDir):
    # again,just test with one image
    # all these partitioned? has 225 submatrices corresponding to each other
    #TODO: implement this for every image in picArray
    #partitionedI = partitionImage(picArray[0])
    #partitionedG = partitionGrad(finalGrad)
    #partitionedD = partitionDir(finalDir)
    # partitioned is a collection of submatrices associated with one imge
    # per elemet in this collection, we have a 10x10 matrix
    gradPerBin = []
    #for i in range(len(partitionedI)):
    #    gradBin(partitionedI[i], finalGrad, finalDir, gradPerBin, k,j)
    # now return complete binList per 
    finalBin = gradBin(picArray, finalGrad, finalDir, gradPerBin)
    return finalBin
    
def main():
    convertToGray()
    picArray = convertToArray()
    #picArray = imagePreProcess()

    #print(picArray[0].shape[0])
    #print(picArray[1].shape[1])
    # now compute magnitude and direction for each pixel
    foFinal = open("output.txt", "w+")
    for i in range(len(picArray)):
        finalGrad1 = getGradient(picArray[i])
        #print(np.isnan(finalGrad1))
        finalDir1 = getMagnitude(picArray[i])
        #print("plz: ", finalGrad1.shape[0])
        #print("plz:", finalGrad1.shape[1])
        #print("plzz: ", finalDir1.shape[0])
        #print("plzz:", finalDir1.shape[1])
        finalBin = hogBin(picArray[i], finalGrad1, finalDir1)
        print("plzzz", len(finalBin))
    #foFinal = open("output.txt", "w+")
        for j in range(len(finalBin)):
            
            foFinal.write(str((finalBin[j])) + ",")
        
        foFinal.write("\n")
        
        #foFinal.close()
main()


