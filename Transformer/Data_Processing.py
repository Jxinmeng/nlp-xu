import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file):
        # 读取源语言和目标语言的文件
        with open(source_file, 'r', encoding='utf-8') as f:
            self.source_sentences = f.readlines()
        with open(target_file, 'r', encoding='utf-8') as f:
            self.target_sentences = f.readlines()

        # 将句子分词或分字，并构建词汇表
        self.source_vocab = self.build_vocab(self.source_sentences)
        self.target_vocab = self.build_vocab(self.target_sentences)

        # 数字化数据
        self.source_data = self.convert_to_index(self.source_sentences, self.source_vocab)
        self.target_data = self.convert_to_index(self.target_sentences, self.target_vocab)

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, index):
        source = self.source_data[index]
        target = self.target_data[index]
        return {'source': source, 'target': target}

    def build_vocab(self, sentences):
        vocab = {'<unk>': 0, '<sos>': 1, '<eos>': 2, '<pad>': 3}
        for sentence in sentences:
            words = sentence.strip().split()  # 使用空格分词，根据实际情况可调整分词方法
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def convert_to_index(self, sentences, vocab):
        data = []
        for sentence in sentences:
            words = sentence.strip().split()  # 使用空格分词，根据实际情况可调整分词方法
            indexes = [vocab.get(word, vocab['<unk>']) for word in words]
            data.append(indexes)
        return data


# 定义数据集文件路径
source_file = 'path/to/source.txt'
target_file = 'path/to/target.txt'

# 创建数据集对象
dataset = TranslationDataset(source_file, target_file)

# 创建数据加载器
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 遍历数据加载器进行训练
for batch in dataloader:
    source = batch['source']
    target = batch['target']
