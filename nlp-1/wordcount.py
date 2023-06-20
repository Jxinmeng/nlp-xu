# encoding: utf-8
import sys
from tqdm import tqdm

# "abc cd" -> ["abc", "", "cd"]


def handle(srcf):
    vocab = {}
    count = 0
    with open(srcf, "rb") as frd:
        for line in frd:
            count = count + 1
    pbar = tqdm(total=count)

    with open(srcf, "rb") as frd:
        for line in frd:
            tmp = line.strip()  # "\r\n"
            if tmp:
                tmp = tmp.decode("utf-8")
                for word in tmp.split():
                    if word:
                        vocab[word] = vocab.get(word, 0) + 1
            pbar.update(1)
    pbar.close()
    return vocab


def save(fname, obj):
    with open(fname, "wb") as fwrt:
        fwrt.write(repr(obj).encode("utf-8"))


def load(fname):
    with open(fname, "rb") as frd:
        tmp = frd.read().strip()
        rs = eval(tmp.decode("utf-8"))
        return rs


if __name__ == "__main__":
    print(sys.argv)
    save(sys.argv[2], handle(sys.argv[1]))
