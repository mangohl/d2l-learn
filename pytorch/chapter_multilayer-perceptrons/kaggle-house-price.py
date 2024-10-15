import hashlib
import os
import tarfile
import zipfile
import requests

# !pip install pandas
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

#url.split('/')[-1] get the filename
#如果文件存在其hash匹配就不用再下载了
def download(name,cache_dir=os.path.join('..','data')):
    assert name in DATA_HUB ,f"{name} not exist in {DATA_HUB}"
    url,sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir,exist_ok=True)
    fname = os.path.join(cache_dir,url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break;
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download from {url} downloading {fname} ...')
    r = requests.get(url,stream=True,verify=True)
    with open(fname,'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name,folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir,ext = os.path.splitext(fname)
    if ext == 'zip':
        fp = zipfile.ZipFile(fname,'r')
    elif ext in ('.tar','gz'):
        fp = tarfile.open(fname,'r')
    else:
        assert False,'only .tar/.zip can unzip'
    fp.extractall(base_dir)
    return os.path.join(base_dir,folder) if folder else data_dir

def download_all():
    for name in DATA_HUB:
        download(name)
        
        
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
#iloc是panda的函数，可以按index索引数据
#print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#1:-1 从第一列到倒数第1列
#all_features包含了所有样本的特征信息
all_features = pd.concat((train_data.iloc[:,1:],test_data.iloc[:,1:]))

#all_features的shape?
#先返回所有类型是数值的特征所在的所有列
#numric_features类型是pandas中Index对象，这里可以理解为列名称
#all_features[]是按列名进行操作
#该步骤是对所有数值特征作归一化预处理
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)


#将object类型(比如字符串)数据进行独热编码
all_features = pd.get_dummies(all_features, dummy_na=True,dtype=int)
all_features.shape


#将panda格式转换为张量
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

#in_features表示有特征数量
loss = nn.MSELoss()
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

#clamp(X,1,float('inf')),将X的值限制在[1,inf]之间
#rmse.item()将只有一个元素的张量转换为标量
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

#nn.MSELoss返回的是均方误差，是求过平均的了，所以不用l.mean().backward()
def train(net,train_features,train_labels,test_features,test_labels,
          num_epochs,lr,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = d2l.load_array((train_features,train_labels),batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


#将X分为K个部分，每个部分大小为X.shape[0] /k，并指定第i部分为验证集
#不够整除时，会导致一部分样本丢失
#k这验证会使用一折作用验证集合，其余的作为训练集
#每次train使用的验证集是train集的0-k中的其中一折
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

#train_ls[-1] get the last value in train_ls
#这里对利用所有样本一共训练了k*epochs次，每次train训练epochs次数
def k_fold(k,X_train,y_train,num_epochs,lr,weight_decay,batch_size):
    train_l_sum,valid_l_sum =0,0
    for i in range(k):
        data = get_k_fold_data(k,i,X_train,y_train)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,lr,weight_decay,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
