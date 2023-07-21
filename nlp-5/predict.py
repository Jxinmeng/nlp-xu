import torch
from model import LanguageModel
import pickle

file_path = "D:/nlp-x/nlp-5/word_to_idx.pkl"

embedding_dim = 32
hidden_dim = 128
output_features = 32


# 使用 pickle 从文件中加载字典
with open(file_path, 'rb') as file:
    word_to_idx = pickle.load(file)

vocab_size = len(word_to_idx)

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
        next_word = idx_to_word.get(next_word_index, "<UNK>")  # 将索引转换为词语
        # 添加预测的词到输出序列中
        result.append(next_word)
    output = "".join(result)
    return output


# 给定的前三个词
w1, w2, w3 = '台','湾','力'
words = [w1,w2,w3]

model = LanguageModel(vocab_size, embedding_dim, hidden_dim, output_features)
# 加载训练好的模型参数
model.load_state_dict(torch.load('D:/nlp-x/nlp-5/model/model_final.pth'))
output = decode_sequence(words,word_to_idx,model)
print(output)