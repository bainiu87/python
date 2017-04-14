# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
apple_number = 100
pear_number = 50

np.random.seed(100)

cov = [[100,0],[0,100]]

#苹果
apple_mean = [0,0]
X_a,Y_a = np.random.multivariate_normal(mean=apple_mean,cov=cov,size=apple_number).T
Z_a = np.repeat(1,apple_number)
#梨
pear_mean = [20,20]
X_p,Y_p = np.random.multivariate_normal(mean=pear_mean,cov=cov,size=pear_number).T
Z_p = np.repeat(0,pear_number)


X = np.append(X_a,X_p)
Y = np.append(Y_a,Y_p)
Z = np.append(Z_a,Z_p)

P_a = list(Z).count(1)/len(Z)
P_p = list(Z).count(0)/len(Z)

#计算区间
def split_range(X, unit_len):
    split_list = []
    X_min = min(X)
    X_max = max(X)
    split_list.append([X_min])
    while True:
        last_min = X_min
        X_min += unit_len
        split_list.append([last_min,X_min])
        if X_min >= X_max:
            break
    return split_list

#计算区间概率
def get_split_p(X,split_list):
    split_p = []
    # split_list = split_range(x,unit_len)
    for i in split_list:
        if len(i) == 1:
            split_p.append((1/len(X)))
        else:
            n = 0
            for l in X:
                if  i[0]< l <=i[1]:
                    n += 1
            if n == 0:
                n += 1
            split_p.append((n/len(X)))
    return split_p

#获得苹果 xy 区间对应的概率 和梨 xy 区间对应的概率
def prior_prob(X_a,Y_a):
    P_x_a_q = split_range(X_a,2)
    P_x_a = get_split_p(X_a,P_x_a_q)
    P_y_a_q = split_range(Y_a,2)
    P_y_a = get_split_p(Y_a,P_y_a_q)
    return [[P_x_a,P_x_a_q],[P_y_a,P_y_a_q]]

P_list_a = prior_prob(X_a,Y_a)
P_list_p = prior_prob(X_p,Y_p)
right_x = []
right_y = []
error_x = []
error_y = []
for l in xrange(0,len(X)):
    for i in xrange(0,len(P_list_a[0][1])):
        if i == 0:
            if X[l] <= P_list_a[0][1][0][0]:
                P_a_x = P_list_a[0][0][i]
        else:
            if P_list_a[0][1][i][0] < X[l] <= P_list_a[0][1][i][1]:
                P_a_x = P_list_a[0][0][i]
    for i in xrange(0,len(P_list_a[1][1])):
        if i == 0:
            if Y[l] <= P_list_a[1][1][0][0]:
                P_a_y = P_list_a[1][0][i]
        else:
            if P_list_a[1][1][i][0] < Y[l] <= P_list_a[1][1][i][1]:
                P_a_y = P_list_a[1][0][i]


    for i in xrange(0,len(P_list_p[0][1])):
        if i == 0:
            if X[l] <= P_list_p[0][1][0][0]:
                P_p_x = P_list_p[0][0][i]
        else:
            if P_list_p[0][1][i][0] < X[l] <= P_list_p[0][1][i][1]:
                P_p_x = P_list_p[0][0][i]
    for i in xrange(0,len(P_list_p[1][1])):
        if i == 0:
            if Y[l] <= P_list_p[1][1][0][0]:
                P_p_y = P_list_p[1][0][i]
        else:
            if P_list_p[1][1][i][0] < Y[l] <= P_list_p[1][1][i][1]:
                P_p_y = P_list_p[1][0][i]
    predicted_a = P_a_x * P_a_y * P_a
    predicted_p = P_p_x * P_p_y * P_p
    if predicted_a > predicted_p:
        if Z[l] == 1:
            right_x.append(X[l])
            right_y.append(Y[l])
        else:
            error_x.append(X[l])
            error_y.append(Y[l])
    else:
        if Z[l] == 0:
            right_x.append(X[l])
            right_y.append(Y[l])
        else:
            error_x.append(X[l])
            error_y.append(Y[l])







plt.figure(figsize=(10,10),dpi=80)
plt.subplot(2,1,1)
plt.plot(X_a,Y_a,".",color="red")
plt.plot(X_p,Y_p,".",color="blue")
plt.subplot(2,1,2)
plt.plot(right_x,right_y,".",color="green",label="right point")
plt.plot(error_x,error_y,".",color="red",label="error point")
plt.show()