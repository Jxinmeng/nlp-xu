import torch
import torch.nn as nn

from pytorch.linear import LinearFunction

class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        # 将张量转换为模块的可训练参数
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)
        # 将权重 weight 初始化为在范围 [-0.1, 0.1] 内均匀分布的值。如果 bias 为 True，则将偏置 bias 初始化为在范围 [-0.1, 0.1] 内均匀分布的值
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return torch.matmul(input, self.weight) + self.bias

