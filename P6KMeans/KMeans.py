# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:52:25 2019

@author: miru, HalfdanJ
"""

import struct
from struct import unpack
import numpy as np
import random

def processDict(generator):
    counter = 0
    imageList = []
    for value in generator:
        if counter < 1000:
            # append the image pixel data or whatever for each image
            imageList.append(value['image'])
        counter += 1
    return imageList

def processNormalize(imgList):
    # first fix m @ 50
    m = 50
    counter = 0
    # init final x and y
    finalX = np.zeros((1000, m))
    finalY = np.zeros((1000, m))
    for spoon in imgList:
        x_cord = np.array( sum( (cord[0] for cord in spoon), () ) )
       
        y_cord = np.array( sum( (cord[1] for cord in spoon), () ) ) 
     
        #print(x_cord.shape)
        # for each of these we have to downsample
        # if len(x_cord) > m we down sample
        if len(x_cord) > m:
            #downX = np.zeros(len(x_cord)
            xPrime = np.zeros(m)
            #downY = np.zeros(len(y_cord))
            yPrime = np.zeros(m)
            for i in range(m):
                roundTo = int(round( (i * len(x_cord)) / (m) ))
                #print("huehue", roundTo)
                xPrime[i] = x_cord[roundTo]
                yPrime[i] = y_cord[roundTo]
        # otherwise,
        else:
            if len(x_cord) < m:
                # init xprime and yprime
                #upsample
                q = (m - 1) / float(len(x_cord) - 1)
                # now check if this is of type int
               # print("q", q)
                if isinstance(q, int):
                    
                    # upsample as usual
                    xPrime = np.zeros(m)
                    yPrime = np.zeros(m)
                    for j in range(m):
                        roundTo = int(np.floor(j / float(q)))
                        
                        if j % q == 0:
                            xPrime[j] = x_cord[roundTo]
                            yPrime[j] = y_cord[roundTo]
                        else:
                            xPrime[j] = ( (1 - ( (j % q) / float(q) ) )*x_cord[roundTo] ) + ( ( (j % q) / float(q) )*x_cord[roundTo + 1] )
                            yPrime[j] = ( (1 - ( (j % q) / float(q) ) )*y_cord[roundTo] ) + ( ( (j % q) / float(q) )*y_cord[roundTo + 1] )
                else:
                    # upsample to ceiling[(q)(len(x) - 1) + 1] and
                    
                    newCeil = int(np.ceil(q))*(len(x_cord) - 1) + 1
                    xPrime = np.zeros(newCeil)
                    yPrime = np.zeros(newCeil)
                    #print("nC", newCeil)
                    for k in range(newCeil):
                        #print(newCeil)
                        roundTo = int(np.floor(k / float(np.ceil(q)) ))
                        #print("qqq", roundTo)
                        if k % np.ceil(q) == 0:
                            xPrime[k] = x_cord[roundTo]
                            yPrime[k] = y_cord[roundTo]
                        else:
                            xPrime[k] = ( (1 - ( (k % np.ceil(q)) / float(np.ceil(q)) ) )*x_cord[roundTo] ) + ( ( (k % np.ceil(q)) / float(np.ceil(q)) )*x_cord[roundTo +1] )
                            yPrime[k] = ( (1 - ( (k % np.ceil(q)) / float(np.ceil(q)) ) )*y_cord[roundTo] ) + ( ( (k % np.ceil(q)) / float(np.ceil(q)) )*y_cord[roundTo + 1] )
                    # downsample to m
                    xDubPrime = xPrime
                    yDubPrime = yPrime
                    xPrime = np.zeros(m)
                    yPrime = np.zeros(m)
                    for l in range(m):
                        roundTo = int(round( (l * newCeil) / float(m) ))
                        #print("sup ", roundTo)
                        xPrime[l] = xDubPrime[roundTo]
                        yPrime[l] = yDubPrime[roundTo]
                    
            else:
                # just set it as how original x and y were
                xPrime = x_cord
                yPrime = y_cord
        finalX[counter] = xPrime
        finalY[counter] = yPrime
        counter += 1
        
    return finalX, finalY

  
def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image,
    }


def unpack_drawings(filename):
    counter = 0
    data = []
    with open(filename, 'rb') as f:
        while counter < 1000:
            try:
                data.append(unpack_drawing(f))
                counter += 1
            except struct.error:
                break
    return data

def MergeXandY(finalX, finalY):
    # (X1,...,Xm, Y1,...,Ym)
    return np.concatenate((finalX, finalY), axis = 1)

def imagesplot(finalX,finalY):
    # we want(X1,Y1,.....,XM,YM) for plotting images.
    return np.stack((finalX,finalY),2).reshape(finalX.shape[0],-1)
    
    
        
def initCentroid(k, finalX, finalY):
    # from 0 -1000, randomly sample k of those and set it to initial centroid indices
    centroidInd = random.sample(range(1000), k)
    centroidInd = np.array(centroidInd)
    centroidX = finalX[centroidInd]
    centroidY = finalY[centroidInd]
  
    return centroidX, centroidY

def updateCM(Centroid, data, k):
      # compute 2 norm residual
      dist = np.subtract(data,Centroid[:, None])
      res = np.linalg.norm(dist, axis = 2)
      # compute argmin along all image axis: 1,....,1000
      ypredict = np.argmin(res, axis=0)
      new_centroid = np.array([np.mean(np.take(data, np.where(ypredict == num)[0], axis = 0), axis = 0) for num in range(k)])
      return new_centroid, ypredict
    
""" wow... finally..."""
def Kmeans(finalX, finalY):
    data = MergeXandY(finalX, finalY)
    #print(data.shape)
    # now generate random K integers
    #k = random.seed(69)
    k = random.randint(5,10)
    
    centroidX, centroidY = initCentroid(k, finalX, finalY)
    Centroid = np.concatenate((centroidX,centroidY), axis = 1)
    
    # initilzlize counter for debugging
    # TODO comment later 
    #counter = 0
    # initialize flag
    converged = False
    meanList = []
   
    #print("bambbbamaba,",  k)
    # we will compute the very first k many clusters' centroids
    first_centroid, ypredict = updateCM(Centroid, data, k)
    new_centroid = first_centroid
    # iterate until convergence
    while converged == False:
        #counter += 1
        Centroid = new_centroid
        # recompute and update
        new_centroid, ypredict = updateCM(Centroid, data, k)
        # check convergence condition. 
        # we now want to do this until the centroids are no longer changing
        # in other words all samples do not change labels any longer
        if np.array_equal(Centroid, new_centroid) == True:
            #print(ypredict)
            #print("size", len(ypredict))
            #print("final itr: ", counter)
            kList =  ypredict
            meanList = new_centroid
            converged = True
            break
        
        #print(counter)
    
    return kList, meanList    

    
def main():
    processed = unpack_drawings('full_binary_spoon.bin')
    midData = processDict(processed)
    #print(type(midData))
    #print(len(midData))
    #print(midData[0])
    finalX,finalY = processNormalize(midData)
    kList, meanList = Kmeans(finalX, finalY)
    print(meanList.shape)
    #print(kList)
    #print(len(kList))
    foFinal = open("output.txt", "w+")
    for i in range(len(kList)):
        foFinal.write(str(int(kList[i])) + "\n")
    foFinal.close()
    zdata = imagesplot(finalX, finalY)
    print(zdata.shape)
    foimage = open("image.txt", "w+")
    #for i in range(2):
    for j in range(100):
        foimage.write(str((zdata[1,j])) + ",")
    foimage.write("\n")
    for j in range(100):
        foimage.write(str((zdata[2,j])) + ",")
    foimage.write("\n")
    for j in range(100):
        foimage.write(str((zdata[3,j])) + ",")
    foimage.close()
    fomean = open("imaZe.txt", "w+")
    for k in range(100):
        fomean.write(str((meanList[1,k])) + ",")
    fomean.write("\n")
    for k in range(100):
        fomean.write(str((meanList[2,k])) + ",")   
    fomean.close()
    
    
            
main()
