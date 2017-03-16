# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

plt.figure(figsize=(10,9),dpi=100)
a_mean = [0,0]
b_mean = [100,100]
cov = [[100,0],
       [0,100]]

#apple
a_X,a_Y = np.random.multivariate_normal(a_mean,cov,1000).T
a_Z = np.repeat(1, 1000)

#peal
p_X,p_Y = np.random.multivariate_normal(b_mean,cov,1000).T
p_Z = np.repeat(-1,1000)

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

#当前B值
def curB(X,Y):
    b_arg = 0
    for i in xrange(0,len(X)):
        b_arg += Y[i]
    b = b_arg / len(X)
    return b

#a、b的偏导数函数
def getGrad(X,Y,Z,cur_a,cur_b,cur_c):
    grad_a = 0
    grad_b = 0
    grad_c = 0
    for i in xrange(0,len(X)):
        grad_a += 2 * X[i] * ( cur_a * X[i] + cur_b * Y[i] + cur_c - Z[i] ) + reg * 2 * cur_a
        grad_b += 2 * Y[i] * ( cur_a * X[i] + cur_b * Y[i] + cur_c - Z[i] ) + reg * 2 * cur_b
        grad_c += 2 * ( cur_a * X[i] + cur_b * Y[i] + cur_c - Z[i] ) + reg * 2 * cur_c
    return [grad_a,grad_b,grad_c]

#初始a、b
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
    a = 0.01

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

plt.subplot(1,1,1)
plt.plot([min(X), max(X)], [min_y, max_y],"r-",label="best line")
plt.plot(a_X, a_Y, "o", color="green", label = "apple")
plt.plot(p_X, p_Y, "o", color="blue", label = "peal" )
plt.legend()

# plt.subplot(2,4,5)
# plt.plot(nmb_list,loss,"ro-",label="loss")
# plt.legend()
#
# plt.subplot(2,4,6)
# plt.plot(nmb_list,grad_a_list,"bo-",label="grad_a")
# plt.legend()
#
# plt.subplot(2,4,7)
# plt.plot(nmb_list,grad_b_list,"yo-",label="grad_b")
# plt.legend()
#
# plt.subplot(2,4,8)
# plt.plot(nmb_list,grad_b_list,"yo-",label="grad_b")
# plt.legend()

plt.show()