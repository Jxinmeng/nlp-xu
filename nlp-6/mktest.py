import h5py
import torch
import pickle

filename = "D:/nlp-x/dataset/tensorfile.hdf5"
filename1 = "D:/nlp-x/dataset/sort.txt"
filename2 = "D:/nlp-x/dataset/word_to_index.pkl"

with open(filename2, 'rb') as file:
    word_to_idx = pickle.load(file)

word_dict_len = len(word_to_idx)


# 使用padding，映射索引为0，填充到相同长度
def pad_sequence(sequence, max_length):
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    return padded_sequence


# 将每个单词映射为索引,并使用pad_sequence函数对indexed_words进行填充操作，并转换为tensor
def convert_to_tensor(batch_data, word_to_index, max_seq_length):
    batch_size = len(batch_data)
    # batch size * seql
    tensor = torch.zeros((batch_size, max_seq_length), dtype=torch.long)

    for i, words in enumerate(batch_data):
        indexed_words = [word_to_index[word] for word in words]
        padded_words = pad_sequence(indexed_words, max_seq_length)
        tensor[i] = torch.tensor(padded_words, dtype=torch.long)

    return tensor


# 根据词典索引，将排序后的数据切分为batch
def handle(filename, word_to_index, m_token):
    max_tokens = 0
    max_seq_length = 0
    batch_data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            seq_length = len(words)
            # 如果当前句子的token数量超过了m_token，,则将之前收集的batch数据yield出去，并清空batch_data列表
            if max_tokens + seq_length > m_token:
                yield convert_to_tensor(batch_data, word_to_index, max_seq_length)
                max_tokens = 0
                max_seq_length = 0
                batch_data = []
            max_tokens += seq_length
            max_seq_length = max(max_seq_length, seq_length)
            batch_data.append(words)
        # 处理剩余的不足一个batch的数据
    if batch_data:
        yield convert_to_tensor(batch_data, word_to_index, max_seq_length)


def save_hdf5_file(filename, data, word_dict_size):
    with h5py.File(filename, "w") as file:
        # 创建一个名为 src 的数据组
        src_group = file.create_group("src")
        for idx, batch in enumerate(data):
            src_group.create_dataset(str(idx), data=batch)
        # ndata: 一个只有一个元素的向量，存src中数据的数量；
        # nword: 一个只有一个元素的向量，存收集的词典大小。
        file.create_dataset("ndata", data=[idx+1])
        file.create_dataset("nword", data=[word_dict_size])
        # 查看src_group中的数据集
        # for name, member in src_group.items():
        #     print(f"成员名称: {name}")
        #     print(f"成员值: {member[()]}")
        #     print("---")


tensor_data = handle(filename1, word_to_idx, m_token=2560)
save_hdf5_file(filename, tensor_data, word_dict_len)

