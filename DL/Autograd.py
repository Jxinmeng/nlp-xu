import torch
x = torch.arange(4.0)
x.requires_grad_(True)
x.grad # 默认值是none

y = 2*torch.dot(x,x)
print(y)


# 通过调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
x.grad
x.grad ==4*x

# 在默认情况下，pytorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad

x.grad.zero_()
y = x * x
y.sum().backword()
x.grad

x.grad.zero_()
y = x * x
u = y.detach()   # u对于系统来讲是一个常数
z = u * x
z.sum().backward()
x.grad == u

def f(a):
    b = a * 2
    while b.norm()<1000:
        b = b * 2
        if b.sum()>0:
            c = b
        else:
            c = 100*b
        return c

a = torch.randn(size=(),requires_grad=True)
d = f(a)
d.backward()

a.grad ==d/a