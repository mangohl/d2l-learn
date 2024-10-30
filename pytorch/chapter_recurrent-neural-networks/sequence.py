import torch
from torch import nn
from d2l import torch as d2l

T = 1000 
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))


#取x[0,996)的数据填充features的第0列
#取x[1,997)的数据填充features的第1列
#取x[2,998)的数据填充features的第2列
#取x[3,999)的数据填充features的第3列
#取x[4,1000)的数据填充lables
#这样下来，features每一行对应一个训练样本,是x[i,i+4)4个时间点的数据
#每个样本对应的标签是x[i+4]
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600

train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)



# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
# 线性层：将4个输入特征映射到10个输出特征
#   特征矩阵大小为10*4,所有样本都共用这一个特征矩阵，不断的更新它
# 预测时输入的X第二个维度要是4
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

#利用features[0,4)预测的是x[4],利用feateures[1,5)预测的是x[5]
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

#先将x[0,604)复制到multistep_preds
#在使用[600,604),[601,605),...net预测后，赋值给multisetp_preds[604],[605],...
#使用[601,605)来预测x[605]是会用到上一个的预测值multisetp_preds[604]作为输入，
#以此类推越往后的预测，会用到更多的预测值作为输入，这将导致预测误差变大

multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))



#对于torch[2:5],其含义是从第2行到第4行，下标都是从0开始，左闭右开
#features[933,68]
#features[:,0]=x[0:933)
#features[:,1]=x[1:934)
#features[:,2]=x[2:935)
#features[:,3]=x[3:936)

'''
以上操作后,features的值如下:
col:0      1      2      3  
    x[0],  x[1],  x[2],  x[3]
    x[1],  x[2],  x[3],  x[4]
    x[2],  x[3],  x[4],  x[5]
    ...
    ...
    x[932],x[934],x[935],x[936]
'''
#features[:,4]=net(features[:,[0,4)]) //result: features[:,4]为预测值
#features[:,5]=net(features[:,[1,5)]) //result: features[:,5]为预测值
#features[:,6]=net(features[:,[2,6)]) //result: features[:,6]为预测值
#features[:,7]=net(features[:,[3,7)]) //result: features[:,7]为预测值
#...
#features[:,67]=net(features[:,[63,67)]) //result: features[:,67]为预测值
#features第4列开始,都是预测值
#
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]


for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

'''
time[4:937),features[:,4],一步预测:预测未来1天的,没有用到预测值
time[7:640),features[:,7],4步骤预测:预测未来4天的,当预测未来第4天时,会用到3个预测值作为输入
time[19:652),features[:,19]
time[67:700),features[:,67]
'''
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))