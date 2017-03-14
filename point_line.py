# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pylab import *
import time
#坐标轴显示范围
plt.xlim(0,110)
plt.ylim(0,110)
X=[ 4.121 , 1.147 , 1.440 , 3.063 ,  2.479 , 1.080
,  2.681 , 3.090 ,  3.718 , 1.501,100]
Y=[ 1.902 , 3.272 , 1.715 , 2.874 ,  1.019 , 1.491
 , 0.939 ,  4.231 , 2.625 , 2.931,100]

#损失函数
def Objvalue(X,Y,cur_a,cur_b):
    su=0
    for i in xrange(0,len(X)):
        x=X[i]
        y=Y[i]
        y_hat=cur_a*x+cur_b
        su+=(y-y_hat)**2
    su=su+cur_a**2+cur_b**2
    return su

#当前B值
def curB(X,Y):
    b_arg=0
    for i in xrange(0,len(X)):
        b_arg+=Y[i]
    b=b_arg/len(X)
    return b

#a、b的偏导数函数
def getGrad(X,Y,cur_a,cur_b):
    grad_a=0
    grad_b=0
    for i in xrange(0,len(X)):
        grad_a += 2 * cur_a * X[i] ** 2 + 2 * cur_b * X[i] - 2 * X[i] * Y[i] + 2 * cur_a
        grad_b += 2 * cur_a * X[i] + 2 * cur_b - 2 * Y[i] + 2 * cur_b
    return [grad_a,grad_b]

#初始a、b
cur_a=0
cur_b=curB(X,Y)

#默认当前期望值和上一次期望值
cur_hope=1e7
last_hope=1e8


#收敛阀值
eplision=1e-06

while abs(last_hope-cur_hope) > eplision:
    print "当前收敛值(0.0001)："+str(abs(last_hope-cur_hope))
    last_hope=cur_hope
    cur_hope=Objvalue(X,Y,cur_a,cur_b)

    #得到a、b偏导数
    grad_all = getGrad(X, Y, cur_a, cur_b)
    grad_a = grad_all[0]
    grad_b = grad_all[1]
    print "A 损失值:"+str(cur_hope)
    a=0.01
    while True:
        next_a=cur_a-a*grad_a
        next_b=cur_b-a*grad_b
        next_grad_all = getGrad(X, Y, next_a, next_b)
        next_grad_a = next_grad_all[0]
        next_grad_b = next_grad_all[1]
        if abs(next_grad_a) >abs(grad_a):
            a = a / 10
        else:
            break
    cur_a-=a * grad_a
    cur_b-=a * grad_b
    print "a、b的偏导数：" + str(grad_a) + "/" + str(grad_b)
cur_c= 100 * cur_a + cur_b
plt.plot([0, 100], [cur_b, cur_c])
scatter(X, Y)
show()

