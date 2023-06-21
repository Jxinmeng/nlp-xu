import math
from collections import Counter

filename1 = "../dataset/newstest2016.tc.en"
filename2 = "../dataset/newstest2016.tc.de"


# 计算 TF
def handle(srcf):
    vocab = {}
    with open(srcf, "rb") as frd:
        for line in frd:
            tmp = line.strip()  # "\r\n"
            if tmp:
                tmp = tmp.decode("utf-8")
                for word in tmp.split():
                    if word:
                        vocab[word] = vocab.get(word, 0) + 1
    count_sum = sum(vocab.values())
    vocab1 = vocab.copy()
    for word, count in vocab1.items():
        vocab1[word] = vocab1[word] / count_sum
    return count_sum, vocab, vocab1


sum_en, vocab_en, tf_en = handle(filename1)
sum_de, vocab_de, tf_de = handle(filename2)

# 计算IDF
X, Y = Counter(vocab_en), Counter(vocab_de)
vocab = dict(X + Y)
count_sum = sum(vocab.values())
idf = vocab.copy()
for word, count in vocab.items():
    idf[word] = - math.log(vocab[word] / count_sum)


# 计算 TFIDF
def tfidf(counts, tf):
    for word, count in counts.items():
        counts[word] = tf[word] * idf[word]
    return counts


tfidf_en = tfidf(vocab_en, tf_en)
tfidf_de = tfidf(vocab_de, tf_de)

# 预测句子是英语还是德语
def predict(s, dic_en, dic_de):
    de_sum = 0
    en_sum = 0
    for word in s.split():
        if word not in dic_en:
            dic_en[word] = 0
        if word not in dic_de:
            dic_de[word] = 0
        en_sum += dic_en[word]
        de_sum += dic_de[word]
    return en_sum, de_sum

# 输入句子得到预测结果
sentence = input()
tmp = predict(sentence, tfidf_en, tfidf_de)
print("en:\n", tmp[0])
print("de:\n", tmp[1])
if tmp[0] >= tmp[1]:
    print("en")
else:
    print("de")

# yesterday this was not the case : the light had barely turned green for pedestrians when a luxury vehicle sped through on a red light .
# Sie prüfen derzeit , wie sie im Laufe der nächsten zehn Jahre zu einem System wechseln können , bei dem Fahrer pro gefahrener Meile bezahlen .
