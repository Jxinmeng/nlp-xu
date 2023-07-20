import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(3 * embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, 32, bias=False),
            nn.Linear(32, vocab_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        embedded = self.embedding(input)
        features = embedded.view(embedded.size(0), -1)
        output = self.fc(features)
        return output

# 读取数据文件
with open('D:\\nlp-x\dataset\\news.2022.zh.shuffled.deduped', 'r', encoding='utf-8') as file:
    data = file.read().split()

# 构建词典
word_to_idx = {word: idx for idx, word in enumerate(data)}
idx_to_word = {idx: word for idx, word in enumerate(data)}
vocab_size = len(word_to_idx)

# 定义训练数据和目标数据
inputs = []
targets = []
for i in range(len(data) - 3):
    inputs.append([word_to_idx[word] for word in data[i:i+3]])
    targets.append(word_to_idx[data[i+3]])

# 创建模型实例
embedding_dim = 32
hidden_dim = 128
model = LanguageModel(vocab_size, embedding_dim, hidden_dim)

# 设置随机种子
torch.manual_seed(0)

# 初始化模型权重
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

# 创建损失函数和优化器
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

# 将模型和损失函数移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_function = loss_function.to(device)

# 将训练数据和目标数据转换为Tensor并移至GPU（如果可用）
inputs = torch.tensor(inputs, dtype=torch.long, device=device)
targets = torch.tensor(targets, dtype=torch.long, device=device)

# 训练模型
cur_batch = 0
for epoch in range(3):
    for i in range(0, len(data) - 3, 10):
        batch_inputs = inputs[i:i+10]
        batch_targets = targets[i:i+10]

        # 前向传播
        output = model(batch_inputs)
        loss = loss_function(output, batch_targets)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_batch += 1
        if cur_batch % 1000 == 0:
            torch.save(model.state_dict(), f'model_{cur_batch}.pth')

    print(f'Epoch {epoch+1} completed.')

# 保存最终模型
torch.save(model.state_dict(), 'final_model.pth')
