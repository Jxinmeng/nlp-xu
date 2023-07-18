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

# 词汇量大小
vocab_size = 10000
# 词向量维度
embedding_dim = 32
# 隐层维度
hidden_dim = 128

# 创建模型
model = LanguageModel(vocab_size, embedding_dim, hidden_dim)

# 设置随机种子
torch.manual_seed(0)

# 初始化模型参数
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

# 创建损失函数
loss_function = nn.NLLLoss(ignore_index=0)  # 忽略padding的loss计算

# 创建优化器
optimizer = optim.Adam(model.parameters())

# 将模型和损失函数移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_function = loss_function.to(device)

# 训练
cur_batch = 0
for batch in data_loader:
    batch = batch.to(device)
    target = batch[:, 3:].contiguous().view(-1)
    inputs = batch[:, :3]

    # 前向传播
    output = model(inputs)
    loss = loss_function(output, target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新参数
    optimizer.step()

    cur_batch += 1
    if cur_batch % 1000 == 0:
        torch.save(model.state_dict(), 'model.pth')

