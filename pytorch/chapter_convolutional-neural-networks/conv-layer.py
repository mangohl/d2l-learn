import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1),X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)

#定义一个卷积层，
#先定义核参数，前向传播中再用其参与互相关运算
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,X):
        return corr2d(X,self.weight) + self.bias
    

X = torch.ones((6, 8))
X[:, 2:6] = 0
X
#这个卷积核可以使的水平相邻的两元素相同，则输出为零，否则输出为非零
#从而可以做到检测垂直边缘
#!!卷积的本质是有效提取相邻像素间的相关特征
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
Y

#利用pytorch的内置类定义一个卷积层
#通过上面定义的X,Y来训练卷积层，最终输出卷积核参数
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X = X.reshape(1,1,6,8)
Y = Y.reshape(1,1,6,7)
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr*conv2d.weight.grad
    if (i+1) % 2 ==0:
        print(f'epoch {i+1},loss {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))