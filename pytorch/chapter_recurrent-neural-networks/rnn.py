

'''
为什么不用前面的网络来解决序列模型？
1.输入输出的长度长度是不固定的
2.

各神经网络用图形表示
'''

%matplotlib inline
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

'''
把0和2生成长度为len(vocab)的one-hot向量
'''
F.one_hot(torch.tensor([0, 2]), len(vocab))

'''
5,2,28
'''
X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape



'''
H(t) = activate{ X(t)W_xh + H(t-1)W_hh + b_h }
O(t) = activate{ H(t)W_hq + b_q }
num_hiddens为隐藏层神经元个数
num_inputs = num_outputs = 独热编码长度
'''
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

'''
初始化H(0)
'''
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

'''
inputs为一个批次的输入数据
(H,)表示只有一个元素的元组
cat(outputs,dim=0),将列表output按行进行拼接

假设之前用seq_data_iter_sequential这个函数获取data_iter
假设之前第一个iter=[
     [1,2,3],
     [5,6,7]
    ]
按照之前seq_data_iter_sequential的实现,这个iter的shape为
(批量大小(2),时间步数量(3))

在下面的函数中可以发现inputs=iter.T
    [
     [1,5],
     [2,6]
     [3,7]
    ]
inputs的shape为(时间步数量,批量大小)
for X in inputs (是按第一维从inputs取数据)
第一次X取的应该是时间步1:[1,5]这两个样本
第二次X取的应该是时间步2:[2,6]这两个样本

'''
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    # H的形状: (批量大小，隐藏层神经元数)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)