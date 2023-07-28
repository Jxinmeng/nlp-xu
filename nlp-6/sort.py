import random

filename = "D:/nlp-x/dataset/train.txt"
output = "D:/nlp-x/dataset/sort.txt"
sentence_list = []

# 读取句子，并将长度低于3或长于256的丢弃
with open(filename, "r", encoding='utf-8') as file:
    for line in file:
        if len(line.split()) > 256 or len(line.split()) < 3:
            continue
        sentence_list.append(line)


# 获取每个句子和以及该句子的token个数，将按token个数倒序放入字典中
def sort(s_list):
    data = {}
    for s in s_list:
        x = len(s.split())
        if x in data:
            data[x].append(s)
        else:
            data[x] = [s]
    sorted_data = dict(sorted(data.items(), key=lambda item: item[0], reverse=True))
    return sorted_data


count_token = sort(sentence_list)

# 将同一长度的句子通过random包的shuffle方法做一次顺序打乱，防止数据收集时相近分布的数据聚集在一起
with open(output, "w", encoding='utf-8') as file:
    for key in count_token:
        random.shuffle(count_token[key])
        for tmp in count_token[key]:
            file.write(tmp)
