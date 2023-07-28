set -e -o pipefail -x
export dir=D:/nlp-x/dataset
export tgtd=D:/nlp-x/dataset

# 设置merge operation
export bpeops=5000
# 设置词频阈值
export minfreq=10
# 创建输出文件目录
mkdir -p $tgtd
# 存储学习到的bpe代码
export output_bpe=$tgtd/bpe.codes

# 在训练集上学BPE  -s设置merge operation的数量，不用添加--min-frequency设置词汇阈值
# --min-frequency用于学习BPE模型时，只有在训练集中出现频率达到该阈值的子词（包括原始单词）才会被合并，这里不需要使用
subword-nmt learn-bpe -s $bpeops  < $dir/output.txt > $output_bpe

# 对训练集初步BPE的结果统计vocabulary，统计结果将保存在 train_vcb.txt 文件中
subword-nmt apply-bpe -c $output_bpe < $dir/output.txt | subword-nmt get-vocab > $tgtd/train_vcb.txt

# 使用学到的BPE和统计的vocabulary对训练集应用BPE,并将结果保存到 train.txt 文件中
# --vocabulary-threshold用于应用BPE时，只有在统计得到的词汇量中出现频率达到该阈值的单词才会被进行BPE处理。
subword-nmt apply-bpe -c $output_bpe --vocabulary $tgtd/train_vcb.txt --vocabulary-threshold $minfreq < $dir/output.txt > $tgtd/train.txt

# 用训练集上学到的BPE和统计的vocabulary对验证集应用BPE,并将结果保存到 dev.txt 文件
subword-nmt apply-bpe -c $output_bpe --vocabulary $tgtd/train_vcb.txt --vocabulary-threshold $minfreq < $dir/output.txt > $tgtd/dev.txt