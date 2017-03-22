# -*- coding: utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot as plt

#求目标函数的导数
def get_grad(cur_a, cur_b, cur_c, X, Y, Z):
    # 1 代表 苹果 -1 代表梨
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
    grad_list = [grad_a_a+grad_p_a,grad_a_b+grad_p_b,grad_a_c+grad_p_c]
    return grad_list

#目标函数
def loss_function(cur_a, cur_b, cur_c, X, Y, Z):
    #1 代表苹果 -1 代表梨
    E = np.e
    P_a = 0
    P_p = 0
    for i in xrange(0,len(X)):
        if Z[i] == 1:
            P_a += cur_a * X[i] + cur_b * Y[i] + cur_c - np.log((1+E ** (cur_a * X[i] + cur_b * Y[i] + cur_c)))
            # print "loss_apple:"+str(P_a)
        else:
            P_p += -np.log((1+E ** (cur_a * X[i] + cur_b * Y[i] + cur_c)))
            # print "loss_pear:" + str(P_p)
    P = P_a + P_p
    return P

# begin
# the number of apple and pear
numb_a = 1000
numb_p = 1000

np.random.seed(10)

cov = [[100,0],[0,100]]

#apple
mean_a = [0,0]
X_a,Y_a = np.random.multivariate_normal(mean=mean_a, cov=cov, size=numb_a).T
Z_a = np.repeat(1, numb_a)

#pear
mean_p = [30,30]
X_p,Y_p = np.random.multivariate_normal(mean=mean_p, cov=cov, size=numb_p).T
Z_p = np.repeat(-1, numb_p)

#merge apple and pear
X = np.append(X_a, X_p)
Y = np.append(Y_a, Y_p)
Z = np.append(Z_a, Z_p)

#初始a b c
cur_a = 2
cur_b = 2
cur_c = 2

#收敛值
eplision = 1e-05

lost_list = []
grad_a_list = []
grad_b_list = []
grad_c_list = []
numb = 0
last_loss = 1e9
cur_loss = 1e8
while abs(last_loss - cur_loss) > eplision and numb < 500:
    numb += 1
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


nmb_list=[]
for i in xrange(1,numb+1):
    nmb_list.append(i)
plt.figure(figsize = (10,20), dpi = 80)

plt.subplot(2,1,1)
plt.plot([min(X),max(X)],[min_y,max_y], "-", color = "black", label="bestLine")
plt.plot(X_a, Y_a, "o", color = "red", label = "apple")
plt.plot(X_p, Y_p, "o", color = "yellow", label = "pear" )
plt.legend()

plt.subplot(2,4,5)
plt.plot(nmb_list,lost_list,"ro-",label="lost")
plt.legend()

plt.subplot(2,4,6)
plt.plot(nmb_list,grad_a_list,"ro-",label="grad_a")
plt.legend()

plt.subplot(2,4,7)
plt.plot(nmb_list,grad_b_list,"ro-",label="grad_b")
plt.legend()

plt.subplot(2,4,8)
plt.plot(nmb_list,grad_b_list,"ro-",label="grad_c")
plt.legend()
plt.show()
