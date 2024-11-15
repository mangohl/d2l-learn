
#lines：嵌套列表 每行例如[ [the time machine by h g wells]，[],...]
#tokens：嵌套列表，每行例如[ ['t', 'h', 'e', ' ', 't', 'i', 'm',....],[],...]
#vocab:是一个字典['t',1],key是一个字符，value是对应的数字
#corpus:一个列表，包含整个文本对应的数字 [1,2,3,4,6,4,...]

def load_corpus_time_machine(max_tokens=-1):  #@save
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)