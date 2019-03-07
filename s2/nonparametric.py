#ノンパラメトリックな確率分布推定

import numpy as np
import matplotlib.pyplot as plt

N=10000
delta=0.001
kukan=10
x=np.arange(-kukan,kukan,delta)

data=np.random.randn(N)

#ヒストグラム推定
plt.hist(data,bins=100)
plt.show()

"""
num_arr=[]
for i in range(int(2*kukan/delta)):
	sec_b=-kukan+delta*i
	sec_n=-kukan+delta*(i+1)
	num=len(np.where((sec_b<=data) & (data<sec_n))[0])
	num_arr.append(num)

plt.plot(x,num_arr)
plt.show()
"""

#ガウスカーネル密度推定

#K近傍法
