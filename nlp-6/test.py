import pickle

import h5py

filename = "D:/nlp-x/dataset/tensorfile.hdf5"
filename2 = "D:/nlp-x/dataset/word_to_index.pkl"

with h5py.File(filename, "r") as file:
    batch = file['ndata'][0]  # batch数量
    vocab_size = file['nword'][0]  # 词典大小

with open(filename2, 'rb') as file:
    word_to_idx = pickle.load(file)

invalid_indices = [idx for idx in word_to_idx.values() if idx >= vocab_size]

print(invalid_indices)
