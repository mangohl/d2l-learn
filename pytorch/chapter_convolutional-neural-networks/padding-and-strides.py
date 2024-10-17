import torch
from torch import nn


#先对X增加(1,1)表示批量大小和通道数，方便参与conv2d运算
#最后返回的Y再把前两维去掉
def comp_conv2d(conv2d,X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

#这里定义了padding为1,加入padding填充，可使输入和输出的shape保持不变
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape

#这里设置了步长为2
#设置步长可以使得输出shape变小
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape


