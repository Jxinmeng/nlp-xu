import torch
from LSTM import LSTMmodel
import pickle

file_path = "D:/nlp-x/dataset/word_to_index.pkl"

input_size = 32
hidden_size = 128
dropout_rate = 0.1

# 使用 pickle 从文件中加载字典
with open(file_path, 'rb') as file:
    word_to_idx = pickle.load(file)

vocab_size = len(word_to_idx)


# 定义解码函数
def decode_sequence(words, word_to_idx, model, max_len=50):
    # 从索引到词语
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    result = []
    result.extend(words)
    for i in range(max_len):
        # 获取词语对应的索引
        tmp = [word_to_idx[word] for word in result]
        # 获取模型输出结果
        input_ = torch.LongTensor(tmp).unsqueeze(dim=0)
        out = model(input_)
        # 获取最后一个的概率
        output = out[0,-1,:]
        # 获取概率最大的词的索引
        next_word_index = torch.argmax(output).item()
        # 将索引转换为词语
        next_word = idx_to_word.get(next_word_index, "<UNK>")  # 将索引转换为词语
        # 添加预测的词到输出序列中
        result.append(next_word)
    output = "".join(result)
    return output


# 输入的词
words = ['部分', '海外', '新闻']

model = LSTMmodel(input_size, hidden_size, dropout_rate, vocab_size + 1)
# 加载训练好的模型参数
model.load_state_dict(torch.load('D:/nlp-x/nlp-7/model/model_final.pth'))
output = decode_sequence(words, word_to_idx, model)
print(output)
