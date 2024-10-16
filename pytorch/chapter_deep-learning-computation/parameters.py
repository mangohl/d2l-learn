import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
net(X)

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

net[2].weight.grad == None

print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

net.state_dict()['2.bias'].data


def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                         nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}',block1())
    return net
#顺序层0为block2,顺序层1是block1
rgnet = nn.Sequential(block2(),nn.Linear(4,1))
rgnet(X)
print(rgnet)
#访问层0中第1个子块中第0层中的bias
rgnet[0][1][0].bias.data


#m.weight.data.abs() >= 5会生成一个和m.weight.data大小相同的bool张量
def my_init(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]


#第3层和第5层使用的是同一个，他们的参数由同一个张量表示
#在反向传播时，来自多个使用位置的梯度会累积到同一个权重的.grad属性上
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)

print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100

print(net[2].weight.data[0] == net[4].weight.data[0])