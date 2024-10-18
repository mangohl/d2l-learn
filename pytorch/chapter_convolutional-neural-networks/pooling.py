import torch
from torch import nn
from d2l import torch as d2l

#单通道的池运算
def pool2d(X,pool_size,mode='max'):
    p_h,p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1,X.shape[1] - p_w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i:i+p_h,j:j+p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i:i+p_h,j:j+p_w].mean()
    return Y



X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
'''
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
'''


X = torch.cat((X, X + 1), 1)
X
'''tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]]])'''
#池大小3X3，输出通道数与输入通道数相同
#没有做累加操作，池单独对输入通道做运算
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
'''
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]]]])
'''