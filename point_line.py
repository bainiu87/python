# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pylab import *

X=[ 4.121 , 1.147 , 1.440 , 3.063 ,  2.479 , 1.080
,  2.681 , 3.090 ,  3.718 , 1.501]
Y=[ 1.902 , 3.272 , 1.715 , 2.874 ,  1.019 , 1.491
 , 0.939 ,  4.231 , 2.625 , 2.931]

#期望函数
def Objvalue(X,Y,cur_a,cur_b):
    su=0
    for i in xrange(0,len(X)):
        x=X[i]
        y=Y[i]
        y_hat=cur_a*x+cur_b
        su+=abs(y-y_hat)
    return su

#当前B值
def curB(X,Y):
    b_arg=0
    for i in xrange(0,len(X)):
        b_arg+=Y[i]
    b=b_arg/len(X)
    return b

#当前a
cur_a=0
cur_b=curB(X,Y)

#默认当前期望值和上一次期望值
cur_hope=1e7
last_hope=1e8

#默认循环次数
max_iter=1000

#增加系数
delta=0.1

last_b_hope=0
#默认增减率
while abs(last_hope-cur_hope) > 1e-05:
    print "误差值(0.0001)："+str(last_hope-cur_hope)
    last_hope=cur_hope
    #max_iter-=1
    cur_hope=Objvalue(X,Y,cur_a,cur_b)
    print "A 期望值:"+str(cur_hope)
    if cur_hope<last_hope:
        cur_a+=delta
    else:
        delta=-delta/10
        print "增减率："+str(delta)
        cur_a+=delta


print "斜率："+str(cur_a)
cur_c=5*cur_a+cur_b
print "期望C："+str(cur_c)
plt.plot([0, 5], [cur_b, cur_c])
scatter(X, Y)
show()

