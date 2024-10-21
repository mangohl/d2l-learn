import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)


num_inputs=784
num_outputs=10

#这里权重矩阵W is 786*10
#每一列对应的是 输出是类i(0-9)所用到的参数
W = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

#.sum(1,) 按行累加？
#该函数使得每一行中的所有列加起来等于1
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition


#X from train_iter is 256*1*28*28
#这里把X reshape成了256*784
#X*W+b:256*10
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W) + b)


# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0, 1], y]
# [0,1]作为行索引，y作为列索引 （高级索引方式）
# 所以是从y_hat中选取第0行0列，和第1行2列 两个值，即0.1和0.5
# 即利用高级索引提取出了对应真实类别的预测概率



#定义交叉熵损失函数
#range(len(y_hat))：生成从 0 到 len-1 的索引
#换句话说，交叉熵损失就是对真实类别的预测概率求对数的负数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])


#计算预测准确的个数
#y_hat = y_hat.argmax(axis=1):返回每一列中最大值的索引
#y_hat.type(y.dtype) == y：判断y_hat是否等于y，返回的cmp为(True,False)
#上面如果不做类型转换如何？

#len(y_hat.shape)返回的是y_hat是几维的，如果y_hat是3x3,返回的是2
#python中只有对可变对象参数进行原地操作时，才会修改外部的参数?
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
       y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self,n):
        self.data = [0.0] * n

    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
#判断net是不是pytorch的神经网络模型
#将模型net设置为评估模式
#y.numel返回y中的元素数量
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel)
    return metric[0] / metric[1]


#l.mean()计算l的平均值，这里即计算平均损失
#l.mean().backward()反向传播是对平均损失来计算梯度
#如果直接l.backward()反向传播时对总损失来计算梯度，即梯度是每个样本的梯度和
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel)
    return metric[0] / metric[2] , metric[1] / metric[2]


class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

#一个epoch会遍历所有的测试集样本
#遍历方法由test_iter迭代决定
#assert train_loss < 0.5, train_loss 当train_loss>0.5时抛异常并打印
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
    train_loss,train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)