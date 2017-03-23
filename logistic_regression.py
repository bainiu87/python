# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt
from operator import itemgetter

"""
    P(apple|(x,y)) / P(pear|(x,y)) = e^(a*x + b*y + c)
    P(apple|(x,y)) + P(pear|(x,y)) = 1
    推导出 P(apple|(x,y)) 和 P(pear|(x,y)) 的方程式

"""
#求目标函数的导数
def get_grad(cur_a, cur_b, cur_c, X, Y, Z):
    # 1 代表 苹果 -1 代表梨 lamda L2正则化超参
    lamda = 0.001
    E = np.e
    grad_a_a = 0
    grad_a_b = 0
    grad_a_c = 0
    grad_p_a = 0
    grad_p_b = 0
    grad_p_c = 0
    for i in xrange(0,len(X)):
        p = 1+E ** (cur_a * X[i] + cur_b * Y[i] + cur_c)
        if Z[i] == 1:
            grad_a_a += X[i] - 1/p * (p - 1) * X[i]
            grad_a_b += Y[i] - 1/p * (p - 1) * Y[i]
            grad_a_c += 1- 1/p * (p - 1)
        elif Z[i] == -1:
            grad_p_a += -1/p * (p - 1) * X[i]
            grad_p_b += -1/p * (p - 1) * Y[i]
            grad_p_c += -1/p * (p - 1)
    grad_list = [grad_a_a + grad_p_a + (lamda * 2 * cur_a),grad_a_b + grad_p_b + (lamda * 2 * cur_b),grad_a_c + grad_p_c + (lamda * 2 * cur_c)]
    return grad_list

#目标函数
def loss_function(cur_a, cur_b, cur_c, X, Y, Z):
    #1 代表苹果 -1 代表梨 lamda L2正则化超参
    E = np.e
    lamda = 0.001
    P_a = 0
    P_p = 0
    for i in xrange(0,len(X)):
        if Z[i] == 1:
            P_a += cur_a * X[i] + cur_b * Y[i] + cur_c - np.log((1+E ** (cur_a * X[i] + cur_b * Y[i] + cur_c)))
            # print "loss_apple:"+str(P_a)
        else:
            P_p += -np.log((1+E ** (cur_a * X[i] + cur_b * Y[i] + cur_c)))
            # print "loss_pear:" + str(P_p)
    # 加入L2 正则化
    P = P_a + P_p + lamda * (cur_a ** 2 + cur_b ** 2 + cur_c ** 2)
    return P

#ROC 中TP FP
def get_T_F_P(numb_a,numb_p,cur_a,cur_b,cur_c,X,Y,Z):
    #Z[i] 真实值 z 预测值
    TP = []
    FP = []
    tp = 0
    fp = 0
    P = []
    for i in xrange(0,len(X)):
        z = cur_a * X[i] + cur_b * Y[i] + cur_c
        p = [Z[i],z]
        P.append(p)
    R = sorted(P,reverse=True,key=itemgetter(1))
    for l in xrange(0,len(R)):
        if R[l][0] == 1:
            tp += 1
            TP.append(tp / (numb_a * 1.000))
            FP.append(fp / (numb_p * 1.000))
        if R[l][0] == -1:
            fp += 1
            TP.append(tp / (numb_a * 1.000))
            FP.append(fp / (numb_p * 1.000))
    return [TP,FP]

#计算auc
def auc(x, y):
    sumarea = 0.
    px, py = 0, 0

    for i in xrange(len(x)):
        sumarea += (py + y[i]) * (x[i] - px) / 2

        px = x[i]
        py = y[i]

    sumarea += (1 + py) * (1 - px) / 2
    return sumarea

# begin
# the number of apple and pear
numb_a = 1000
numb_p = 1000

np.random.seed(10)

cov = [[10,0],[0,10]]

#apple
mean_a = [0,0]
X_a,Y_a = np.random.multivariate_normal(mean=mean_a, cov=cov, size=numb_a).T
Z_a = np.repeat(1, numb_a)

#pear
mean_p = [3,3]
X_p,Y_p = np.random.multivariate_normal(mean=mean_p, cov=cov, size=numb_p).T
Z_p = np.repeat(-1, numb_p)

#merge apple and pear
X = np.append(X_a, X_p)
Y = np.append(Y_a, Y_p)
Z = np.append(Z_a, Z_p)

#初始a b c
cur_a = 0.02
cur_b = 0.02
cur_c = 0.02

#收敛值
eplision = 1e-05

lost_list = []
grad_a_list = []
grad_b_list = []
grad_c_list = []
nmb_list = []
numb = 0
last_loss = 1e9
cur_loss = 1e8
while abs(last_loss - cur_loss) > eplision and numb < 500:

    numb += 1
    nmb_list.append(numb)
    print "number："+str(numb)

    last_loss = cur_loss
    cur_loss = loss_function(cur_a,cur_b,cur_c,X,Y,Z)

    print "realEplision:"+str(last_loss - cur_loss)
    #将损失值添加到列表
    lost_list.append(cur_loss)

    grad_list = get_grad(cur_a, cur_b, cur_c, X, Y, Z)

    grad_a = grad_list[0]
    grad_a_list.append(abs(grad_a))

    grad_b = grad_list[1]
    grad_b_list.append(abs(grad_b))

    grad_c = grad_list[2]
    grad_c_list.append(abs(grad_c))

    #控制系数
    control = 0.001/numb
    while True:
        next_a = cur_a + control * grad_a
        next_b = cur_a + control * grad_b
        next_c = cur_c + control * grad_c
        next_grad_list = get_grad(next_a,next_b,next_c,X,Y,Z)
        next_grad_a = next_grad_list[0]
        next_grad_b = next_grad_list[1]
        next_grad_c = next_grad_list[2]
        if abs(next_grad_a) > abs(grad_a) and abs(next_grad_b) > abs(grad_b) and abs(next_grad_c) > abs(grad_c) and control > 1e-08:
            control /= 10
        else:
            break
    print "control:"+str(control)
    cur_a += control * grad_a
    cur_b += control * grad_b
    cur_c += control * grad_c

b = -cur_c / cur_b
max_y = -max(X) * (cur_a/cur_b) + b
min_y = -min(X) * (cur_a/cur_b) + b

# ROC 中的 TP 和 FP
P_all = get_T_F_P(numb_a,numb_p,cur_a,cur_b,cur_c,X,Y,Z)
TP = P_all[0]
FP = P_all[1]

AUC=auc(FP, TP)
print "AUC:"+str(AUC)

plt.figure(figsize = (10,20), dpi = 80)

plt.subplot(3,1,1)
plt.plot([min(X),max(X)],[min_y,max_y], "-", color = "black", label="bestLine")
plt.plot(X_a, Y_a, "o", color = "red", label = "apple")
plt.plot(X_p, Y_p, "o", color = "yellow", label = "pear" )
plt.legend()

plt.subplot(3,4,5)
plt.plot(nmb_list, lost_list, "ro-", label="lost")
plt.legend()

plt.subplot(3,4,6)
plt.plot(nmb_list, grad_a_list, "bo-", label="grad_a")
plt.legend()

plt.subplot(3,4,7)
plt.plot(nmb_list, grad_b_list, "yo-", label="grad_b")
plt.legend()

plt.subplot(3,4,8)
plt.plot(nmb_list, grad_c_list, "go-", label="grad_c")
plt.legend()

plt.subplot(3,1,3)
plt.plot(FP, TP, "ro",label="ROC")
plt.legend()

plt.show()
