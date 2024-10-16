import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
net(X)


#继承nn.Module
#初始化和重写forward函数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
net(X)

#继承nn.Module,自定义顺序架构
#利用内置的_models变量存储用户输入的层参数
#forward函数依次调用内部层
class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            self._modeles[idx] = module

    def forward(self,X):
        for block in self.__modules.values():
            X = block(X)
        return X
    

net = MySequential(nn.Linear(20,256),nn.ReLu(),nn.Linear(256,10))
net(X)


#加入了常量参数rand_weight和自定义函数参与到forward的输出
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand(20,20),requires_grad=False
        self.linear = nn.Linear(20,20)

    def forward(self,X):
        X = self.linear(X)
        X = F.relu(torch.mm(X,self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X/=2
        return X.sum()
    
net = FixedHiddenMLP()
net(X)


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))
    
chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
chimera(X)

