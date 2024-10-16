from mxnet import np, npx
from mxnet.gluon import nn

npx.set_np()

##这里使用的是另外一个框架，
#256代表的是输出维度，这里没有指定输入维度，框架会自动确认
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()