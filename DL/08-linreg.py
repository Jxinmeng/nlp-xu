import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 创建合成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 构造一个pytorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# 定义模型和损失函数
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()

# 定义随机梯度下降优化器，用于更新模型参数
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l = loss(net(x),y)
        # 梯度清零
        trainer.zero_grad()
        # 反向传播和参数更新
        l.backward()
        trainer.step()

    # 计算当前epoch的损失
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')