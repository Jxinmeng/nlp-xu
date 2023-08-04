import torch
import torch.nn as nn
import torch.optim as optim
from LSTM import LSTMmodel
# from RNN import RNNmodel
import h5py
from tqdm import tqdm


filename = "D:/nlp-x/dataset/tensorfile.hdf5"
file_path = "D:/nlp-x/nlp-7/model/model_final.pth"

input_size = 32
hidden_size = 128
dropout_rate = 0.1
epoch = 30

with h5py.File(filename, "r") as file:
    batch = file['ndata'][0]  # batch数量:138
    vocab_size = file['nword'][0]  # 词典大小：9042


# 训练模型
def train(filename, optm, model, loss_f, my_device=None):
    with h5py.File(filename, "r") as file:
        batch = file['ndata'][0]  # batch数量
        for i in range(epoch):
            for t in tqdm(range(batch)):
                # 取每个batch的数据
                seq_batch = torch.from_numpy(file["src"][str(t)][()])
                # 取出除去最后一个词之外的全部词
                input_ = seq_batch[:, :-1]
                # 取出除去第一个词之外的全部词
                target = seq_batch[:, 1:]
                if my_device:
                    input_, target = input_.to(my_device), target.to(my_device)
                # 前向传播
                output = model(input_)
                # output:(batch_size*seq_len,vocab_size), target: batch_size*seq_len
                target_flat = target.flatten()
                loss = loss_f(output.view(-1, vocab_size + 1), target_flat)
                # 反向传播和参数更新
                optm.zero_grad()
                loss.backward()
                optm.step()
                torch.save(model.state_dict(), file_path)


model = LSTMmodel(input_size, hidden_size, dropout_rate, vocab_size + 1)
# model = RNNmodel(input_size, hidden_size, dropout_rate, vocab_size + 1)

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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_function = loss_function.to(device)

train(filename, optimizer, model, loss_function, device)