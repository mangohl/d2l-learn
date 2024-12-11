

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

#返回的lines每一行都是一条文本内容
#每行如下
#the time machine by h g wells
#twinkled and his usually pale face was flushed and animated the
#
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])


#返回一个由词源列表组成的列表
#每一行列表中的词元是字符串
#['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
#[]
#...
#['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

#将所有词元组成的语料库tokens来构建词表
#返回的vocab可通过它来确定词元所对应的索引
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])


#lines：嵌套列表 每行例如[ [the time machine by h g wells]，[],...]
#tokens：这里的tokens是按字符拆分的
#   嵌套列表，每行例如[ ['t', 'h', 'e', ' ', 't', 'i', 'm',....],[],...]
#vocab:是一个字典['t',1],key是一个字符，value是对应的数字
#corpus:一个列表，包含整个文本对应的数字[1,2,3,4,6,4,...],即文本按照字符的数字化版本
#len(vocab)这里会返回不同字符的数量
def load_corpus_time_machine(max_tokens=-1):  #@save
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')#字符词元进行分割
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)