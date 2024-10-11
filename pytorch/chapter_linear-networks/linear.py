import random
import torch
from d2l import torch as d2l



def synthetic_data(w,b,num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.001,y.shape)
    return X,y.reshape((-1,1))

#true_w是一维张量，不区分行列,它的shape是(3,)
#而X的shape是(1000,3),所以X * w可行
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),
                labels.detach().numpy(),
                1)

#num_examples-1?
#why tensor?
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield features[batch_indices],labels[batch_indices]


batch_size=10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

#normal x?y?
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def linreg(X,w,b):
    return torch.matmul(X,w)+b

def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

# grad.zero放到外面也一样？
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()


lr=0.03
num_epochs=3
net=linreg
loss=squared_loss

#why l.sum().backward()?
#if not,sgd dose not need / batch_size?
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')