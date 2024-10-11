import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

#返回一个迭代器，每次返回batch_size个样本
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)

#2:一个样本情况下的特征数（1个样本的输入维度
#1：一个样本的输出维度
#定义了一个一层神经网络，这一层是线性层(全连接层)
net = nn.Sequential(nn.Linear(2,1))

#初始化参数
net[0].weight.data.normal_(0,0.01)
net[0].weight.data.fill_(0)

#定义损失函数
loss = nn.MSELoss()

#定义优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1},loss {l:f}')


w = net[0].weight.data
print('w的估计误差:',true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:',true_b - b)