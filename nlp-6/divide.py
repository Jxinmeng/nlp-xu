import jieba

filename = "D:/nlp-x/dataset/news.2010.zh.shuffled.deduped"
output = "D:/nlp-x/dataset/output.txt"

with open(filename, 'r', encoding='utf-8') as f1:
    with open(output, 'w', encoding='utf-8') as f2:
        for line in f1:
            # 使用jieba进行分词，默认精确模式
            seg_list = list(jieba.cut(line.strip(), cut_all=False))
            # 将分词结果转换为字符串并打印输出
            if len(seg_list) <= 128:
                f2.write(" ".join(seg_list))
                f2.write("\n")

