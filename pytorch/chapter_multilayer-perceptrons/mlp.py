import torch
from torch import nn
from d2l import torch as d2l

batch_size  = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#num_hiddens是隐藏层的神经元的个数
num_inputs,num_outputs,num_hiddens = 784,10,256

W1 = nn.Parameter(torch.randn(
    num_inputs,num_hiddens,requires_grad=True)*0.01)
b1 = nn.Parameter(torch.zeros(
    num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

param = [W1,b1,W2,b2]

#若X([-1,2]
#    [-3,5])
#则relu(X)返回([0,2]
#              [0,5])
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X@W1+b1)
    return (H@W2+b2)

#'node'表示损失函数不会求平均或者求和
loss = nn.CrossEntropyLoss(reduction='none')

num_epochs,lr = 10,0.1
updater = torch.optim.SGD(param,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs)

d2l.predict_ch3(net,test_iter)