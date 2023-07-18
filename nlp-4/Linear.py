import torch
from torch import nn
from torch.autograd import Function


# 自定义函数类LinearFunction，它继承了torch.autograd.Function类，并实现了自定义函数的具体计算逻辑
class LinearFunction(Function):
    @staticmethod
    # 前向传播，接收输入input、权重weight和偏置bias作为参数，并返回计算得到的输出。
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    # 设置上下文信息，在反向传播计算之前保存输入和输出等相关信息。接收ctx对象、输入参数和前向传播的输出作为参数
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    # 反向传播，它接收ctx对象和反向传播中的梯度输出grad_output作为参数，并计算得到输入的梯度。
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        # grad_output为反向传播上一级计算得到的梯度值
        grad_input = grad_weight = grad_bias = None
        # 计算input的梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        # 计算weight的梯度
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # 表示需要计算bias的梯度，则执行 grad_output.sum(0),得到与bias相同维度的梯度
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# 将自定义函数类LinearFunction封装在函数linear中
linear = LinearFunction.apply


class Linearmodule(nn.Module):
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
        nn.init.uniform_(self.weight, -0.01, 0.01)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, input):
        return torch.matmul(input, self.weight.t()) + self.bias


net_module = Linearmodule(4,6)
input = torch.randn(10,4)
output = net_module(input)
print(output.size())


class SiLU(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)

# example
input = 4
output = 6

linear_module = Linearmodule(input,output)
switch_activation = SiLU()
x = torch.randn(1,input)
output = linear_module(x)
switch_output =switch_activation(output)
print(output)
print(switch_output)