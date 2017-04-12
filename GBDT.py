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

plt.figure(figsize = (10,10),dpi = 80)
plt.subplot(211)

plt.plot(X_a,Y_a,'o',c='red')
plt.plot(X_p,Y_p,'o')
#合并
X = np.append(X_a,X_p)
Y = np.append(Y_a,Y_p)
Z = np.append(Z_a,Z_p)

#获取label 平均值函数
def get_label_avg(Z):
    avg = 0
    for i in Z:
        avg += i
    label = avg/(len(Z)*1.0)
    return label

#mse求值函数
def entropy(Z):
    result = 0
    if len(Z) == 0:
        return 0
    else:
        z_avg = sum(Z) / (len(Z) * 1.0)
        for i in Z:
            result += (i - z_avg)**2
        return result

#返回分裂规则
def split(X, Y, Z, R):
    best_split_x = ""
    best_entropy = 1e-19
    best_split_y = ""
    for split_i_x in X:
        left_x = []
        right_x = []
        n_x = 0
        for cur_i_x in X:
            if cur_i_x <= split_i_x:
                left_x.append(Z[n_x])
            else:
                right_x.append(Z[n_x])
            n_x += 1
        R_l_x = entropy(left_x)
        R_r_x = entropy(right_x)
        cur_entropy_diff_x = R - R_l_x - R_r_x
        if cur_entropy_diff_x > best_entropy:
            best_split_x = split_i_x
            best_entropy = cur_entropy_diff_x

    for split_i_y in Y:
        left_y = []
        right_y = []
        n_y = 0
        for cur_i_y in Y:
            if cur_i_y <= split_i_y:
                left_y.append(Z[n_y])
            else:
                right_y.append(Z[n_y])
            n_y += 1
        R_l_y = entropy(left_y)
        R_r_y = entropy(right_y)
        cur_entropy_diff_y = R - R_l_y - R_r_y
        if cur_entropy_diff_y > best_entropy:
            best_split_y = split_i_y
            best_entropy = cur_entropy_diff_y
    if best_split_y == "":
        return ["x",best_split_x]
    else:
        return ["y",best_split_y]

#label 平均值
label_avg_1 = get_label_avg(Z)
Z_label = Z - label_avg_1
#根节点
R = entropy(Z_label)
best_split = split(X, Y, Z_label, R)
print "分割点"+str(best_split)
X_l_1 = []
Y_l_1 = []
Z_l_1 = []
X_r_1 = []
Y_r_1 = []
Z_r_1 = []
if best_split[0] == "x":
    for i in xrange(0,len(X)):
        if X[i] <= best_split[1]:
            X_l_1.append(X[i])
            Y_l_1.append(Y[i])
            Z_l_1.append(Z_label[i])
        else:
            X_r_1.append(X[i])
            Y_r_1.append(Y[i])
            Z_r_1.append(Z_label[i])
else:
    for i in (0,len(Y)):
        if Y[i] <= best_split[1]:
            X_l_1.append(X[i])
            Y_l_1.append(Y[i])
            Z_l_1.append(Z_label[i])
        else:
            X_r_1.append(X[i])
            Y_r_1.append(Y[i])
            Z_r_1.append(Z_label[i])

left_avg = get_label_avg(Z_l_1)
right_avg = get_label_avg(Z_r_1)
root_avg = get_label_avg(Z_label)
print "左边："+str((root_avg+left_avg))
print "右边："+str((root_avg+right_avg))
numb_l = len(Z_l_1)
numb_r = len(Z_r_1)
plt.subplot(212)

right_x = []
right_y = []
wrong_x = []
wrong_y = []

if (root_avg + left_avg) > 0.5:
    for i in xrange(0,numb_l):
        if Z_l_1[i] == (1 - label_avg_1):
            right_x.append(X_l_1[i])
            right_y.append(Y_l_1[i])
        else:
            wrong_x.append(X_l_1[i])
            wrong_y.append(Y_l_1[i])
    for i in xrange(0,numb_r):
        if Z_r_1[i] == (0 - label_avg_1):
            right_x.append(X_r_1[i])
            right_y.append(Y_r_1[i])
        else:
            wrong_x.append(X_r_1[i])
            wrong_y.append(Y_r_1[i])
else:
    for i in xrange(0,numb_l):
        if Z_l_1[i] == (0 - label_avg_1):
            right_x.append(X_l_1[i])
            right_y.append(Y_l_1[i])
        else:
            wrong_x.append(X_l_1[i])
            wrong_y.append(Y_l_1[i])
    for i in xrange(0,numb_r):
        if Z_r_1[i] == (1 - label_avg_1):
            right_x.append(X_r_1[i])
            right_y.append(Y_r_1[i])
        else:
            wrong_x.append(X_r_1[i])
            wrong_y.append(Y_r_1[i])

plt.plot(right_x,right_y,'o', c='g')
plt.plot(wrong_x,wrong_y,'o', c='r')
plt.show()
