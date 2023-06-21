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

# 保存在文件中
tfidf_en = tfidf(vocab_en, tf_en)
tfidf_de = tfidf(vocab_de, tf_de)

with open("tfidf_en.txt","w",encoding="utf-8") as file:
    for word, count in tfidf_en.items():
        file.write("%s %lf\n" % (word, count))

with open("tfidf_de.txt","w",encoding="utf-8") as file:
    for word, count in tfidf_de.items():
        file.write("%s %lf\n" % (word, count))