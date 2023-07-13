import torch
from torch.autograd import Function


class LinearFunction(Function):
    # 被声明为静态方法，因此可以通过类直接调用，而不需要创建类的实例。
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
