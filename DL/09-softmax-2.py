import torch
from IPython import display
from d2l import torch as d2l

# 每次随机读取256张图片
batch_size = 256
# 返回训练集和测试集的迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition


def net(x):
    # 256*28*28
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)


# 实现交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])



def accuracy(y_hat, y):  # @save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 评估在任意模型 net 的准确率
def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for x, y in data_iter:
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


# Accumulator实例中创建了 2 个变量，用于分别储存正确预测的数量和预测的总数量
class Accumulator:  # @save
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

print(evaluate_accuracy(net, test_iter))
