# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt

#the number of apple and pear
numb_a = 100
numb_p = 100

np.random.seed(10)

cov = [[100,0],[0,100]]

#apple
mean_a = [0,0]
X_a,Y_a = np.random.multivariate_normal(mean=mean_a, cov=cov, size=numb_a).T
Z_a = np.repeat(1, numb_a)

#pear
mean_p = [50,50]
X_p,Y_p = np.random.multivariate_normal(mean=mean_p, cov=cov, size=numb_p).T
Z_p = np.repeat(-1, numb_p)

#merge apple and pear
X = np.append(X_a, X_p)
Y = np.append(Y_a, Y_p)

#loss function









plt.figure(figsize = (10,20), dpi = 80)
plt.plot(X_a, Y_a, "o", color = "red", label = "apple")
plt.legend()
plt.plot(X_p, Y_p, "o", color = "yellow", label = "pear" )
plt.legend()

plt.show()
