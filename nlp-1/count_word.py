filename = "../dataset/newstest2016.tc.en"
counts = {}

with open(filename, encoding="utf-8",errors="ignore") as file:
    for line in file:
        words = line.split()
        for word in words:
            counts[word] = counts.get(word, 0) + 1

# 将结果保存在文件中
with open("count_word.txt","w",encoding="utf-8") as file:
    for word, count in counts.items():
        file.write("%s %d\n" % (word, count))
