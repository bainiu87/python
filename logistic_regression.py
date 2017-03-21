# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt

#求目标函数的导数
def get_log(a, b, c, X, Y, Z):
    # 1 代表 苹果 -1 代表梨
    E = np.e
    log_a_a = 0
    log_a_b = 0
    log_a_c = 0
    log_p_a = 0
    log_p_b = 0
    log_p_c = 0
    for i in xrange(0,len(X)):
        p = E ** (a * X[i] + b * Y[i] + c)
        if Z[i] == 1:
            log_a_a += X[i] - 1/p * (p - 1) * X[i]
            log_a_b += Y[i] - 1/p * (p - 1) * Y[i]
            log_a_c += 1- 1/p * (p - 1)
        elif Z[i] == -1:
            log_p_a += -1/p * (p - 1) * X[i]
            log_p_b += -1/p * (p - 1) * Y[i]
            log_p_c += -1/p * (p - 1)
    log_numb = [[log_a_a,log_a_b,log_a_c],[log_p_a,log_p_b,log_p_c]]
    return log_numb

#目标函数
def loss_function(a, b, c, x, y, z):
    #1 代表苹果 -1 代表梨
    E = np.e
    P_a = 0
    P_p = 0
    if z == 1:
        P_a += a * x + b * y + c - np.log(1+E ** (a * x + b * y + c))
    else:
        P_p += -np.log(1+E ** (a * x + b * y + c))
    P = P_a + P_p
    return P

# begin
# the number of apple and pear
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
Z = np.append(Z_a, Z_p)

#初始a b c
a = 1
b = 2
c = 3

log_numb=get_log(a,b,c,X,Y,Z)

#收敛值
eplision = 1e-05

last_loss = 1e8
cur_loss = 1e7

# while abs(last_loss - cur_loss) > eplision:



plt.figure(figsize = (10,20), dpi = 80)
plt.plot(X_a, Y_a, "o", color = "red", label = "apple")
plt.legend()
plt.plot(X_p, Y_p, "o", color = "yellow", label = "pear" )
plt.legend()

#plt.show()
