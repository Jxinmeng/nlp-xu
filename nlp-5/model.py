import torch.nn as nn


# 创建语言模型
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_features):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_features, bias=False),
            nn.Linear(output_features, vocab_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        embedded = self.embedding(input)
        # 将嵌入向量的维度进行变换，从而将其展平为一个特征向量
        features = embedded.view(embedded.size(0), -1)
        output = self.net(features)
        return output
