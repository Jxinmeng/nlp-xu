import torch
import torch.nn as nn
import torch.optim as optim
from model import LanguageModel
import numpy as np

filename = "D:\\nlp-x\\dataset\\news.2016.zh.shuffled.deduped"


# 计算文件中字的个数
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

# 构建字典
word_to_idx = {word: idx for idx, word in enumerate(vocab)}


# 训练模型
def train(word_to_idx, data, optm, model, loss_f, my_device=None):
    curb = 0
    for batch in data:
        inputs = []
        length = len(batch)
        # 使用每个batch中的词生成输入序列和目标序列
        if length <= 3:
            continue
        targets_words = batch[3:length]
        targets_index = [word_to_idx[word] for word in targets_words]
        for i in range(3):
            inputs_words = batch[i:length - 3 + i]
            inputs_index = [word_to_idx[word] for word in inputs_words]
            inputs.append(inputs_index)
        array = np.array(inputs)
        input = torch.tensor(array).transpose(0, 1)
        input = input.long()
        target = torch.tensor(targets_index)
        if my_device:
            input, target = input.to(my_device), target.to(my_device)
        # 前向传播
        output = model(input)
        loss = loss_f(output, target)
        # 反向传播和参数更新
        optm.zero_grad()
        loss.backward()
        optm.step()
        # 每10000个batch保存一次模型
        curb += 1
        if curb % 10000 == 0:
            file_path = f'D:\\nlp-x\\nlp-5\\model\\model_{curb}.pth'
            torch.save(model.state_dict(), file_path)
    torch.save(model.state_dict(), 'D:\\nlp-x\\nlp-5\\model\\model_final.pth')

# 创建模型实例
embedding_dim = 32
hidden_dim = 128
output_features = 32
model = LanguageModel(vocab_size, embedding_dim, hidden_dim, output_features)

# 设置随机种子
torch.manual_seed(0)

# 模型参数初始化
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

# 创建损失函数和优化器
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

# 如果可以使用GPU，将模型和损失函数移到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_function = loss_function.to(device)

train(word_to_idx, data, optimizer, model, loss_function, device)
