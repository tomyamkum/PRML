#ベイズ推定を使って,一次関数の係数を推測する

import numpy as np
import matplotlib.pyplot as plt

#N:データ数
N=20

#一次関数:y=a0+a1x+epsilon
#epsilon:ノイズ
a0=-0.3
a1=0.5

#mu:事前分布の平均,alpha:事前分布の精度パラメータ
mu=np.array([0,0])
alpha=2.0

#beta:ノイズの精度パラメータ
beta=(1/0.2)**2

#事前分布
w0=np.arange(-1,1,0.01)
w1=np.arange(-1,1,0.01)

W0,W1=np.meshgrid(w0,w1)

Z=np.sqrt(alpha/(2*np.pi))*np.exp(-alpha/2*(W0-mu[0])**2)*np.sqrt(alpha/(2*np.pi))*np.exp(-alpha/2*(W1-mu[1])**2)
Z=Z/np.sum(Z)
plt.pcolormesh(W0,W1,Z)
plt.title("確率分布")
plt.legend()
plt.show()

#逐次推定
x_arr=[]
t_arr=[]
for n in range(N):
	#x:[-1,1]の一様分布
	x=np.random.rand()*2-1

	#epsilon:ガウスノイズ
	epsilon=np.random.normal(0,np.sqrt(1/beta))

	x_arr.append(x)
	t_arr.append(a0+a1*x+epsilon)

	#尤度関数
	likelihood_arr=[]

	likelihood_arr=np.exp(-beta/2*(t_arr[-1]-W0-W1*x_arr[-1])**2)
	likelihood_arr=np.array(likelihood_arr)/np.sum(likelihood_arr)
	Z=Z*likelihood_arr
	Z=Z/np.sum(Z)
	plt.scatter(x_arr,t_arr)
	plt.title("得られたデータ")
	plt.xlim([-1,1])
	plt.ylim([-1,1])
	plt.legend()
	plt.show()
	plt.pcolormesh(W0,W1,likelihood_arr)
	plt.title("尤度")
	plt.legend()
	plt.show()
	plt.pcolormesh(W0,W1,Z)
	plt.title("確率分布")
	plt.legend()
	plt.show()
