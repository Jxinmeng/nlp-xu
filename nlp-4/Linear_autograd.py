import torch
from torch.autograd import Function


# 自定义函数类LinearFunction，它继承了torch.autograd.Function类，并实现了自定义函数的具体计算逻辑
class LinearFunction(Function):
    @staticmethod
    # 前向传播，接收输入input、权重weight和偏置bias作为参数，并返回计算得到的输出。
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        # unsqueeze(0) 将偏置向量 bias 的维度从 (N,) 扩展为 (1, N)，其中 N 是偏置向量的长度
        # 将扩展维度后的偏置向量按照输出张量 output 的形状进行广播，使其具有相同的形状。
        # 将广播后的偏置向量与输出张量进行逐元素相加。
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
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
        grad_input = grad_weight = grad_bias = None
        # 需要计算输入的梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        # 需要计算权重的梯度
        if ctx.needs_input_grad[1]:
            # grad_output 转置后与输入矩阵 input 的矩阵乘法结果表示了反向传播的梯度传递
            grad_weight = grad_output.t().mm(input)
        # 表示需要计算偏置的梯度，则执行 grad_output.sum(0) 计算偏置项的梯度，即对输出梯度进行求和
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# 将自定义函数类LinearFunction封装在函数linear中
linear = LinearFunction.apply


def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)

