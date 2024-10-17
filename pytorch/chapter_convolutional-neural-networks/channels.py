import torch
from d2l import torch as d2l


#多通道的卷积计算
#将对应通道的x和k进行卷积运算得到y,sum做的是y0+y1+y2
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))