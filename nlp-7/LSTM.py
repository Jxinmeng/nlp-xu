import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTMCell, self).__init__()
        # 输入门
        self.input_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )
        # 遗忘门
        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )
        # 输出门
        self.output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()
        )
        # 根据act计算h
        self.update_cell = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, _input, pre_cell_state, pre_hidden_state):
        # 拼接_input和pre_hidden
        combined = torch.cat((_input, pre_hidden_state), dim=-1)
        # 计算输入门
        input_gate = self.input_gate(combined)
        # 计算遗忘门
        forget_gate = self.forget_gate(combined)
        # 计算输出门
        output_gate = self.output_gate(combined)
        # 计算图中的h
        update = self.update_cell(combined)
        # 更新细胞状态和隐藏状态
        cell_state = forget_gate * pre_cell_state + input_gate * update
        hidden_state = output_gate * pre_hidden_state

        return cell_state, hidden_state


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, vocab_size):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        # 忽略填充符号
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.lstmcell = LSTMCell(input_size, hidden_size, dropout_rate)
        # 初始化第一个时间步的隐藏状态和细胞状态
        self.h_0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        batch_size, seq_length = inputs.size()
        # 嵌入操作
        embedded = self.embedding(inputs)
        # 初始化隐藏状态和细胞状态,扩展为 batch_size 行
        cell_state = self.c_0.expand(batch_size, -1)
        hidden_state = self.h_0.expand(batch_size, -1)
        outputs = []
        # 循环遍历每个时间步
        for t in range(seq_length):
            # 获取当前时间步的输入
            _input = embedded[:, t]
            # 更新隐藏状态和细胞状态
            cell_state, hidden_state = self.lstmcell(_input, cell_state, hidden_state)
            outputs.append(hidden_state)

        # 输出 outputs:(batch_size，seq_len,vocab_size)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.out(outputs)
        return outputs
