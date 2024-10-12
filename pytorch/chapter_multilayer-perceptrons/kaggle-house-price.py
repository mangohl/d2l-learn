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
def download(name,cache_dir=os.path.jon('..','data')):
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
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

#all_features的shape?
#先返回所有类型是数值的特征所在的所有列
#numric_features类型是pandas中Index对象，这里可以理解为列名称
#all_features[]是按列操作
#该步骤是对所有数值特征作归一化预处理
numeric_features = all_features.dyptes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

#将object类型(比如字符串)数据进行独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)
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
