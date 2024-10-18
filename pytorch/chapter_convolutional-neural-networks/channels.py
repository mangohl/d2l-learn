import torch
from d2l import torch as d2l


#多通道的卷积计算
#通道Xi和卷积核Ki进行卷积运算得到y,sum做的是y0+y1+y2
#X与K是同维的，比如都是3维
#输出是1通道的
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

#K比X多一维,所以有多输出通道
#比如X(2,3,3)表示X有2个输入通道，每个通道是2X2的矩阵
#K(3,2,2,2)表示有3个卷积核，每个卷积核对应的输入有2个通道，每个通道是2X2的矩阵
#最终会得到3个通道的输出
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,K) for k in K],0)





'''
# 输入张量:2个通道,大小为3x3
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

# 卷积核:3个卷积核,每个核对应的输入是2个通道,每个卷积核大小为1x1
K = torch.tensor([[[[0.1]], [[0.2]]],
                  [[[0.3]], [[0.4]]],
                  [[[0.5]], [[0.6]]]])

X after reshape:
 tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.]])

K after reshape:
 tensor([[0.1000, 0.2000],
        [0.3000, 0.4000],
        [0.5000, 0.6000]])
'''
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X = X.reshape((c_i,h*w))
    K = K.reshape((c_o,c_i))
    Y = torch.matmul(K,X)
    return Y.reshape((c_o,h,w))