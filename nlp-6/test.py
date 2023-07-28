import h5py

filename = "D:/nlp-x/dataset/tensorfile.hdf5"

with h5py.File(filename, 'r') as h5f:
    dataset_names =list(h5f.keys())
    print("数据集名称:", dataset_names)

    # 选择一个数据集进行查看
    dataset_name = dataset_names[0]
    dataset = h5f[dataset_name]

    # 检查数据集的形状和数据类型
    print("数据集形状:", dataset.shape)
    print("数据集数据类型:", dataset.dtype)

    # 获取数据集的值
    data = dataset[:]
    print("数据集的值:", data)