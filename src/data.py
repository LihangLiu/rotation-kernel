import os

import numpy as np
from torch.utils.data import Dataset


class ModelNet(Dataset):
    def __init__(self, data_path, split, voxel_size):
        self.data_path = data_path
        self.split = split
        self.voxel_size = voxel_size

        self.data = self.load_data(os.path.join(self.data_path, '{0}-{1}.txt'.format(self.split, self.voxel_size)))

    def load_data(self, split_path):
        data = []
        for line in open(split_path, 'r'):
            data_path, category = line.strip().split()
            data.append((os.path.join(self.data_path, data_path), np.int(category)))
        return data

    def __getitem__(self, index):
        data_path, target = self.data[index]

        data = np.load(data_path)
        x = data[:, 0].astype(int)
        y = data[:, 1].astype(int)
        z = data[:, 2].astype(int)

        input = np.zeros((1, self.voxel_size, self.voxel_size, self.voxel_size))
        input[0, x, y, z] = 1

        return input, target

    def __len__(self):
        return len(self.data)
