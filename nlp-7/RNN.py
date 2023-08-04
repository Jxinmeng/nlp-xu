import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(RNNCell, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),dim=-1)
        output = self.net(combined)
        return output


class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, vocab_size):
        super(RNNmodel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, input_size, padding_idx=0)
        self.hidden_size = hidden_size
        # 初始化第一个时间步的隐藏状态
        self.h_0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.cell = RNNCell(input_size, hidden_size, dropout_rate)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.Logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        batch_size, seq_len = inputs.size()
        # 嵌入操作
        embedded = self.embedding(inputs)
        # 初始化隐藏状态,扩展为 batch_size 行
        hidden = self.h_0.expand(batch_size, -1)
        outputs = []

        # 循环遍历每个时间步
        for t in range(seq_len):
            # 获取当前时间步的输入
            input_t = embedded[:,t]
            # 得到输出
            output_t = self.cell(input_t,hidden)
            outputs.append(output_t)
            # 更新隐藏状态
            hidden = output_t

        # 输出 output:(batch_size，seq_len,vocab_size)
        outputs = torch.stack(outputs , dim=1)
        outputs = self.out(outputs)
        outputs = self.Logsoftmax(outputs)
        return outputs