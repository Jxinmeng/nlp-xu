def handle(s, tfidf_en, tfidf_de):
    de_sum = 0
    en_sum = 0
    for word in s.split():
        if word not in tfidf_en:
            tfidf_en[word] = 0
        if word not in tfidf_de:
            tfidf_de[word] = 0
        en_sum += tfidf_en[word]
        de_sum += tfidf_de[word]
    return en_sum, de_sum


def load(fname):
    with open(fname, "rb") as frd:
        tmp = frd.read().strip()
        rs = eval(tmp.decode("utf-8"))
        return rs


if __name__ == "__main__":
    sentence = input()
    dic_en = load("tfidf_en.txt")
    print(1)
    dic_de = load("tfidf_de.txt")
    print(1)
    tmp = handle(sentence, dic_en, dic_de)
    print("en:%f\n", tmp[0])
    print("de:%f\n", tmp[1])
    if tmp[0] >= tmp[1]:
        print("en")
    else:
        print("de")
