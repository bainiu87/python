# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
numb_a = 50
numb_p = 100

np.random.seed(10)

cov = [[10,0],[0,10]]

#apple
mean_a = [0,0]
X_a,Y_a = np.random.multivariate_normal(mean=mean_a ,cov=cov , size=numb_a).T
Z_a = np.repeat(1,numb_a)

#pear
mean_p = [10,10]
X_p,Y_p = np.random.multivariate_normal(mean=mean_p, cov=cov, size=numb_p).T
Z_p = np.repeat(0,numb_p)

#合并
X = np.append(X_a,X_p)
Y = np.append(Y_a,Y_p)
Z = np.append(Z_a,Z_p)

#获取label 平均值函数
def get_label_avg(Z):
    avg = 0
    for i in Z:
        avg += i
    label = avg/(len(Z)*0.1)
    return label

label_avg_1 = get_label_avg(Z)
#根节点熵
R = -((numb_a / ((numb_a + numb_p) * 0.1)) * np.log(numb_a / ((numb_a + numb_p) * 0.1)) + (numb_p / ((numb_a + numb_p) * 0.1)) * np.log(numb_p / ((numb_a + numb_p) * 0.1)))


plt.figure(figsize = (10,10),dpi = 80)
plt.plot(X_a, Y_a, "o", color = "red", label = "apple")
plt.plot(X_p, Y_p, "o", color = "blue", label = "pear")
plt.legend()


#plt.show()