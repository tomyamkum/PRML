#ベイズ曲線フィッティングを行う

import numpy as np
import matplotlib.pyplot as plt

alpha=5.0e-3
beta=11.1
M=9

x_num=10
x=np.arange(0,1,0.01)
sample_x=np.arange(0,1,1/x_num)
get_y=np.sin(2*np.pi*sample_x)+np.random.normal(0,1/11.1,x_num)
t=np.sin(2*np.pi*sample_x)

#plt.scatter(sample_x,t)
#plt.show()

S_1=alpha*np.eye(M)
for i in range(x_num):
	x_row=np.array([])
	for j in range(M):
		x_row=np.append(x_row,sample_x[i]**j)
	x_row=np.array([x_row])
	S_1=S_1+beta*x_row*x_row.T

S=np.linalg.inv(S_1)

def phi(x):
	arr=np.array([])
	for i in range(M):
		arr=np.append(arr,x**i)
	return np.array([arr])

def mean(x):
	sum1=0
	for i in range(x_num):
		sum1=sum1+phi(sample_x[i])*t[i]
	arr=phi(x[0])
	for i in range(len(x)-1):
		arr=np.r_[arr,phi(x[i+1])]
	return beta*np.dot(np.dot(arr,S),sum1.T).T[0]

def s2(x):
	arr=phi(x[0])
	for i in range(len(x)-1):
		arr=np.r_[arr,phi(x[i+1])]
	s=1/beta+np.dot(np.dot(arr,S),arr.T)
	ss=np.array([])
	for i in range(len(x)):
		ss=np.append(ss,s[i][i])
	return ss

yy=mean(x)
sigma=np.sqrt(s2(x))
y_1=yy+sigma
y__1=yy-sigma
plt.plot(x,yy)
plt.fill_between(x,y_1,y__1,facecolor='r',alpha=0.1)
plt.plot(x,np.sin(2*np.pi*x))
plt.scatter(sample_x,t)
plt.show()
