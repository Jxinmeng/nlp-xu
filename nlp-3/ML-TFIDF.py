import random
import math

filename_en = "../dataset/newstest2016.tc.en"
filename_de = "../dataset/newstest2016.tc.de"

w_en = {}
w_de = {}
vocab = set()


# 将文本中出现的单词放入集合中
def count_vocab(srcf):
    vocab1 = set()
    with open(srcf, "rb") as f:
        for line in f:
            tmp = line.strip()
            if tmp:
                tmp = tmp.decode("utf-8")
                for word in tmp.split():
                    if word not in vocab1:
                        vocab1.add(word)
    return vocab1


vocab_en = count_vocab(filename_en)
vocab_de = count_vocab(filename_de)

# 将英语和德语文本中出现的所有单词放入一个集合中
vocab = vocab_en.union(vocab_de)


# 随机初始化每个单词属于各个类别的权重：random_en和random_de
def random_vocab(vocab1):
    random_en = {}
    random_de = {}
    rang = math.sqrt(1 / len(vocab))
    for w in vocab1:
        random_en[w] = random.uniform(-rang, rang)
        random_de[w] = random.uniform(-rang, rang)
    return random_en, random_de


w_en, w_de = random_vocab(vocab)

# 依次取文本中的一行，作为训练集中的一条数据，英语和德语交替读取
def get_line(srcf, tag):
    with open(srcf, "rb") as f:
        for line in f:
            tmp = line.strip()
            if tmp:
                tmp = tmp.decode("utf-8")
            yield tmp, tag


# 根据loss函数，对于训练集中的每一条数据，调整模型参数
def handle_loss(tmp, tag):
    lr = 1e-3
    tmp = tmp.split()
    global w_en, w_de
    score_en = sum(w_en[w] for w in tmp)
    score_de = sum(w_de[w] for w in tmp)
    if tag == "en":
        if score_en < score_de: # loss = max(0, score_de-score_en)
            for w in tmp:
                w_en[w] += lr
                w_de[w] -= lr
    if tag == "de":
        if score_de < score_en:
            for w in tmp:
                w_en[w] -= lr
                w_de[w] += lr


# 均匀读取训练数据，得到模型参数
def handle(srcf1, srcf2):
    # 得到生成器，get_en,get_de
    get_en = get_line(srcf1, "en")
    get_de = get_line(srcf2, "de")
    while True:
        try:
            tmp_en, tag_en = next(get_en)
        except:
            tmp_en = None
        if tmp_en is not None:
            handle_loss(tmp_en, tag_en)
        try:
            tmp_de, tag_de = next(get_de)
        except:
            tmp_de = None
        if tmp_de is not None:
            handle_loss(tmp_de, tag_de)
        if tmp_en is None and tmp_de is None:
            break


# 预测
def predict(s, dic_en, dic_de):
    de_sum = 0
    en_sum = 0
    for word in s.split():
        if word in dic_en:
            en_sum += dic_en[word]
        if word in dic_de:
            de_sum += dic_de[word]
    return en_sum, de_sum


handle(filename_en, filename_de)
sentence = input()
tmp = predict(sentence, w_en, w_de)
print("en:\n", tmp[0])
print("de:\n", tmp[1])
if tmp[0] >= tmp[1]:
    print("en")
else:
    print("de")

# yesterday this was not the case : the light had barely turned green for pedestrians when a luxury vehicle sped through on a red light .
# Sie prüfen derzeit , wie sie im Laufe der nächsten zehn Jahre zu einem System wechseln können , bei dem Fahrer pro gefahrener Meile bezahlen .
