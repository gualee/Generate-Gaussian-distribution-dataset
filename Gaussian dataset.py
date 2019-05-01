# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:37:19 2018
@author: 段昊
"""
import numpy as np
import matplotlib.pyplot as plt

#=======================================================calculation covariance matrix
def cal_cov(dimension,p,c):
    for j in range(dimension):
        c[0][j] = pow(p, j)
    
    for i in range(dimension - 1):
        for j in range(dimension - 1):
            if i == j:
                c[i][j] = 1
            c[i+1][j+1] = c[i][j]
    
    for i in range(dimension):
        for j in range(dimension):
            if i < j:
                c[j][i] = c[i][j]
    return c
#=======================================================parameter setting
N = 100                                                 #data 100 samples
dimension = 20                                          #dimension is 20
mean = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]        #mean vector
p1 = 0.9
p2 = 0.5
cov1 = np.zeros((20, 20))                               #create first zero matrix
cov2 = np.zeros((20, 20))                               #create second zero matrix
cal_cov(dimension,p1,cov1)                              #calculate covariance matrix1
cal_cov(dimension,p2,cov2)                              #calculate covariance matrix2
#=======================================================generation
dataset1 = np.random.multivariate_normal(mean, cov1, N) #generate dataset1
dataset2 = np.random.multivariate_normal(mean, cov2, N) #generate dataset2
print("First dataset(p1 is 0.9): \n", dataset1)         #print dataset1
print("Second dataset(p2 is 0.5): \n", dataset2)        #print dataset1
np.savetxt('D:\\dataset1.csv', dataset1, fmt='%1.4f', delimiter=',') #save to csv file
np.savetxt('D:\\dataset2.csv', dataset2, fmt='%1.4f', delimiter=',') #save to csv file
#=======================================================presentation 
plt.scatter(dataset1[:,0], dataset1[:,1], c='blue')
plt.title('Class1')
X, Y = np.meshgrid(dataset1[:,0], dataset1[:,1])
Z = (X - Y)/2
plt.contour(X,Y,Z, colors='black') 
plt.show()

plt.scatter(dataset2[:,0], dataset2[:,1], c='red')
plt.title('Class2')
X, Y = np.meshgrid(dataset2[:,0], dataset2[:,1])
Z = (X - Y)/2
plt.contour(X,Y,Z, colors='black') 
plt.show()
