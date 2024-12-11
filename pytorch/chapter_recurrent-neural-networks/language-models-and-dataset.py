import random
import torch
from d2l import torch as d2l


'''
这里的token是按word进行拆分的
corpus是一个将这个文本中所有word组合的列表
vocab看作是一个字典[word,int]
'''
tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]

#这里把pytorch的索引下标看作是0,1,2,..和...-3,-2,-1,0(从右开始的下表)
#zip将两个切片中对应位置的元素进行配对，返回一个迭代器
#这一行代码就完成了把corpus中相邻两个word作为一个元组
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]

'''
相邻三个构成一个元组triple
'''
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]



'''
corpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
batch_size = 2//每个批次有2组样本
num_steps = 3//每个样本中的子序列的长度为num_steps,即每组中有num_steps个元素
可以有 ( len((corpus)-1) // batch_size // num_steps ) = num_batchs个批次

initial_indices_per_batch = [3, 0]
一个批次：
X = [[4, 5, 6], [1, 2, 3]]
Y = [[5, 6, 7], [2, 3, 4]]#Y是X后移一位


这个函数的作用是把corpus先分为n组,再打乱每组,每次迭代返回batch_size组

'''
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    #从[0,num_steps-1]中生成一个随机数，并从这个随机数位置截断到尾
    corpus = corpus[random.randint(0, num_steps - 1):]

    # 减去1，是因为我们需要考虑标签
    # 截断后的corpus一共有num_subseqs个num_steps
    num_subseqs = (len(corpus) - 1) // num_steps

    #每组数据的起始索引[0,num_steps,num_steps*2,...] 
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    random.shuffle(initial_indices)

    #从起始位置pos开始，返回一组数据，长度为num_steps
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        #每一个批次中有batch_size组数据
        #每组数据的起始位置[i到i+batch_size）
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


'''
每个批次有batch_size个样本
每个样本中子序列的长度为num_steps个元素
-1为了给Y预留一个位置
//batch_size * batch*size保证可用的语料可以生成完整的批次
假设Xs=[1,2,3,4,5,6,7,8],Yx=[2,3,4,5,6,7,8,9]
batch_size=2;reshape后:
Xs=[[1,2,3,4],
    [5,6,7,8]
    ]
假设num_steps=2,第一次迭代(第一个小批量,第一批次):
X = [[1,2],
     [5,6]
    ]

第二次迭代:
X = [[3,4],
     [7,8]
    ]
可用看出：两个相邻的小批量中的子序列在原始序列上也是相邻的
[:,i:i+num],可用看作是竖着切,间隔为num


这个函数的作用是把corpus先分为batch_size行,再按列间隔为num_steps切开排好
每次迭代返回num_steps列
'''
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y