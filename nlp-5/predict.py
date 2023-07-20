import numpy as np
import torch
from model import LanguageModel
import numpy as np
from torch import nn

filename = "D:\\nlp-x\\dataset\\news.2016.zh.shuffled.deduped"

embedding_dim = 32
hidden_dim = 128
output_features = 32

def count_vocab(srcf):
    vocab = set()
    data = []
    # 读取数据文件
    with open(srcf, "r", encoding="utf-8") as f:
        for line in f:
            tmp = line.strip()
            data.append(tmp)
            if tmp:
                for word in tmp:
                    if word not in vocab:
                        vocab.add(word)
    vocab_size = len(vocab)
    return data, vocab, vocab_size


data, vocab, vocab_size = count_vocab(filename)

word_to_idx = {word: idx for idx, word in enumerate(vocab)}


# 定义解码函数
def decode_sequence(words,word_to_idx, model, max_len=50):
    # 从索引到词语
    idx_to_word = {word: idx for idx, word in word_to_idx.items()}
    result = []
    result.extend(words)
    for i in range(max_len):
        # 获取词语对应的索引
        tmp = [word_to_idx.get(word, 0) for word in result[-3:]]
        # 获取模型输出结果
        input_ = torch.LongTensor(tmp).unsqueeze(dim=0)
        out = model(input_)
        # 获取概率最大的词的索引
        next_word_index = torch.argmax(out).item()
        # 将索引转换为词语
        next_word = idx_to_word.get(next_word_index + 1, "<UNK>")  # 将索引转换为词语
        # 添加预测的词到输出序列中
        result.append(next_word)
    output = "".join(result)
    return output


# 给定的前三个词
w1, w2, w3 = '台','湾','力'
words = [w1,w2,w3]

model = LanguageModel(vocab_size, embedding_dim, hidden_dim, output_features)
# 加载训练好的模型参数
model.load_state_dict(torch.load('D:\\nlp-x\\nlp-5\\model\\model_final.pth'))
output = decode_sequence(words,word_to_idx,model)
print(output)
