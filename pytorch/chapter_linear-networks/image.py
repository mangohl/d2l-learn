import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

#定义一个trans对象，用来将fashionMNIST里的图像转换为张量
#mins_train里的每一张图像都是(image,label)元组
#image的shape是(1,28,28),label是0-9中的一个标量
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data",train=True,transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data",train=False,transform=trans,download=True)


#输入labels = [0, 1, 2, 7]
#返回['t-shirt', 'trouser', 'pullover', 'sneaker']
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#_表示接下来不再使用的返回值
#整个图像是num_rows*num_cols个网格组成
#每个网格是一个子图，用来容纳显示图像数据
#axes.flatten()将子图数组展开为一维的
#zip 将两个列表配对后返回一个迭代器，每个元素是一个元组
#ax.imshow(img) 使用子图控制显示img
#enumerate 返回迭代对象中每个元素和索引
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize = (num_cols*scale,num_rows*scale)
    _, axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
    
    
    
#DataLoader 可以从 mnist_train 数据集中按批次加载 18 个样本。
#返回的是一个可迭代对象
#iter() 将 DataLoader 转换为迭代器。
#next() 提取下一个批次的数据。
#X 包含 18 张图片，y 包含 18 个对应的标签。
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));