# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:50:59 2019

@author: miru
"""
import time
import numpy as np 

x = np.random.rand(1000000)
y = np.random.rand(1000000)

start1 = time.time()
z = np.dot(x,y)
end1 = time.time()

i = 0

start2 = time.time()
for i in range(1000000):
    i += x[i]*y[i]
end2 = time.time()

totalTime1 = end1 - start1
totalTime2 = end2 - start2

#print("vectorization is " + str( (totalTime2 / float(totalTime1) )) + " times faster.")


x = 1
print(x)

x = np.log(x)
print(x)

X = np.array([[1,1,1], [2,2,2], [3,3,3]])
Y = X[:,0]
Z = X[:,1:X.shape[1]]
print(Y)
print(Z)

a = np.array([1,2,3])

b = np.dot(a,X)
c = np.dot(a.T,X)

print(b)
print(c)