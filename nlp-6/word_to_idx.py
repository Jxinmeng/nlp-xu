import pickle
word_index = "D:/nlp-x/dataset/word_to_index.pkl"
filename = "D:/nlp-x/dataset/sort.txt"
word_freq = {}

with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        for word in line.split():
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

# 将其转换为单词到索引的一对一映射
word_to_idx = {word: idx for idx, word in enumerate(word_freq,1)}


# 使用 pickle 将字典保存到文件中
with open(word_index, 'wb') as file:
    pickle.dump(word_to_idx, file)