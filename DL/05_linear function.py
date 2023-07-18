import torch
x = torch.arange(4)
print(x)
print(len(x))

A = torch.arange(20).reshape(5,4)
print(A)
print(A.T)

B = torch.arange(24).reshape(2,3,4)
print(B)

# 通过分配新内存，将A的一个副本分配给B
B = A.clone()
A.sum(axis=0)
A.mean(axis=0)

sum_A = A.sum(axis=1,keepdim=True)
A/sum_A

# 累加求和
A.cumsum(axis=0)

# 向量的点积
torch.dot(A,B)

# 向量积
torch.mm(A,B)

# 范数：向量或矩阵的长度
torch.norm(A)
# L1范数：向量元素的绝对值之和
torch.abs(A).sum()
# F范数
torch.norm(torch.ones((4,9)))