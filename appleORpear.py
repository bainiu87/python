# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import axes3d

#设置图片参数
plt.figure(figsize=(10,20),dpi=100)

#苹果和梨的个数
apple_numb=1000
pear_numb=1000

#mean 高斯分布（正态分布）的中心点）
a_mean = [0,0]
b_mean = [15,15]

#cov 高斯分布（正态分布）的离散程度
cov = [[100,0],
       [0,100]]

#apple （高斯分布点坐标）
a_X,a_Y = np.random.multivariate_normal(a_mean,cov,apple_numb).T
a_Z = np.repeat(1, apple_numb)

#pear （高斯分布点坐标）
p_X,p_Y = np.random.multivariate_normal(b_mean,cov,pear_numb).T
p_Z = np.repeat(-1,pear_numb)

#将apple 和 pear 坐标合并
X = np.append(a_X , p_X)
Y = np.append(a_Y , p_Y)
Z = np.append(a_Z , p_Z)

reg = 0

#loss function
def Objvalue(X, Y, Z, cur_a, cur_b, cur_c):
    su = 0
    for i in xrange(0,len(X)):
        x = X[i]
        y = Y[i]
        z = Z[i]
        z_hat = cur_a * x + cur_b * y + cur_c
        su += (z - z_hat) ** 2
    su = su + reg * (cur_a ** 2 + cur_b ** 2 + cur_c ** 2)
    return su

# 当前B值  最小二乘法，初期确定b值的方法，（局部合理，全局不是最优）
# def curB(X,Y):
#     b_arg = 0
#     for i in xrange(0,len(X)):
#         b_arg += Y[i]
#     b = b_arg / len(X)
#     return b

#a、b、c的偏导数函数
def getGrad(X,Y,Z,cur_a,cur_b,cur_c):
    grad_a = 0
    grad_b = 0
    grad_c = 0
    for i in xrange(0,len(X)):
        grad_a += 2 * X[i] * ( cur_a * X[i] + cur_b * Y[i] + cur_c - Z[i] ) + reg * 2 * cur_a
        grad_b += 2 * Y[i] * ( cur_a * X[i] + cur_b * Y[i] + cur_c - Z[i] ) + reg * 2 * cur_b
        grad_c += 2 * ( cur_a * X[i] + cur_b * Y[i] + cur_c - Z[i] ) + reg * 2 * cur_c
    return [grad_a,grad_b,grad_c]

#ROC 曲线中TP FP值
def get_T_F_P(apple_numb,pear_numb,cur_a,cur_b,cur_c,X,Y,Z):
    P = []
    TP=[]
    FP=[]
    tp = 0
    fp = 0
    for l in xrange(0,len(X)):
        #分类器Z[l] 代表真实； z 为分类器计算的预测值
        z = cur_a * X[l] + cur_b * Y[l] + cur_c
        p=[Z[l],z]
        P.append(p)
    R=sorted(P, reverse=True,key=itemgetter(1))
    for i in xrange(0, len(R)):
        if R[i][0] == 1:
            tp += 1
            TP.append(tp / (apple_numb * 1.000))
            FP.append(fp / (pear_numb * 1.000))
        elif R[i][0] == -1:
            fp += 1
            TP.append(tp / (apple_numb * 1.000))
            FP.append(fp / (pear_numb * 1.000))
    print R
    return [TP,FP]

#计算auc
def get_auc(x,y):
    x_1=x[0]
    y_1=y[0]
    node_place=[]
    for i in xrange(0,len(x)):
        try:
            if x[i+1] != x_1 and y[i+1] !=y_1:
                x_1=x[i]
                y_1=y[i]
                node_place.append(i)
        except:
            node_place.append(len(x)-1)
    area = 0
    for l in xrange(0,len(node_place)):
            if l == len(node_place)-1:
                break
            else:
                area += (y[node_place[l]]+y[node_place[l+1]])*(x[node_place[l+1]]-x[node_place[l]])/2.0
    if y[0] != 0:
        area += (y[0]+y[node_place[0]]) * x[node_place[0]]/2.0
    return area
#初始a、b、c
cur_a = 1
cur_b = 2
cur_c = 3

#默认当前期望值和上一次期望值
cur_hope = 1e7
last_hope = 1e8


#收敛阀值
eplision = 1e-06

#循环次数
nmb = 0
#损失值list
loss = []
#导数a list
grad_a_list = []
#导数b list
grad_b_list = []
#导数c list
grad_c_list = []

while abs(last_hope-cur_hope) > eplision and nmb < 200:
    nmb += 1
    print "当前收敛值(0.0001)：" + str(abs(last_hope - cur_hope))
    last_hope = cur_hope
    cur_hope = Objvalue(X , Y, Z, cur_a, cur_b, cur_c)

    #将当前损失值加入到list
    loss.append(cur_hope)

    #得到a、b、c偏导数
    grad_all = getGrad(X, Y, Z, cur_a, cur_b, cur_c)

    #获去a的导数，将导数加入列表
    grad_a = grad_all[0]
    grad_a_list.append(abs(grad_a))

    #获取b的导数，将导数加入列表
    grad_b = grad_all[1]
    grad_b_list.append(abs(grad_b))

    #获取c的导数，将导数加入列表
    grad_c = grad_all[2]
    grad_c_list.append(abs(grad_c))

    print "损失值:" + str(cur_hope)

    #控制系数
    a = 0.01/nmb

    #获取最佳控制系数a
    while True:
        next_a = cur_a - a * grad_a
        next_b = cur_b - a * grad_b
        next_c = cur_c - a * grad_c
        next_grad_all = getGrad(X, Y, Z, next_a, next_b, next_c)
        next_grad_a = next_grad_all[0]
        next_grad_b = next_grad_all[1]
        next_grad_c = next_grad_all[2]

        if abs(next_grad_a) > abs(grad_a) and abs(next_grad_b) > abs(grad_b) and abs(next_grad_c) > abs(grad_c):
            a = a / 10
        else:
            break

    #梯度下降
    cur_a -= a * grad_a
    cur_b -= a * grad_b
    cur_c -= a * grad_c

    print "a、b、c的偏导数：" + str(grad_a) + "/" + str(grad_b) + "/" + str(grad_c)
    print "cur_a:" + str(cur_a) + "/ cur_b:" + str(cur_b) + "/ cur_c:" + str(cur_c)



b=-cur_c/cur_b
max_y=-max(X) * (cur_a/cur_b)+b
min_y=-min(X) * (cur_a/cur_b)+b

nmb_list=[]
for i in xrange(1,nmb+1):
    nmb_list.append(i)

#获取TP 和 FP
P_all=get_T_F_P(apple_numb,pear_numb,cur_a,cur_b,cur_c,X,Y,Z)
TP=P_all[0]
FP=P_all[1]
AUC=get_auc(FP,TP)
print TP
print FP

print "AUC="+str(AUC)
plt.subplot(3,1,1)
plt.plot([min(X), max(X)], [min_y, max_y],"r-",label="best line")
plt.plot(a_X, a_Y, "o", color="green", label = "apple")
plt.plot(p_X, p_Y, "o", color="blue", label = "pear" )
plt.legend()

plt.subplot(3,4,5)
plt.plot(nmb_list,loss,"ro-",label="loss")
plt.legend()

plt.subplot(3,4,6)
plt.plot(nmb_list,grad_a_list,"bo-",label="grad_a")
plt.legend()

plt.subplot(3,4,7)
plt.plot(nmb_list,grad_b_list,"yo-",label="grad_b")
plt.legend()

plt.subplot(3,4,8)
plt.plot(nmb_list,grad_c_list,"go-",label="grad_c")
plt.legend()

plt.subplot(3,1,3)
plt.plot(FP, TP, "ro-",label="ROC")
plt.legend()
plt.show()

