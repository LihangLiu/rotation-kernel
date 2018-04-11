import os
from os.path import dirname, join
import numpy as np
from torch.utils.data import Dataset


class ModelNet(Dataset):
    def __init__(self, data_path, voxel_size = 32):
        self.data_path = data_path
        self.voxel_size = voxel_size

        self.data = self.load_data(self.data_path)

    def load_data(self, path):
        data = []
        for line in open(path, 'r'):
            a, b = line.strip().split()
            data.append((join(dirname(path), a), b))
        return data

    def __getitem__(self, index):
        path, target = self.data[index]

        points = np.load(path)
        xs = points[:, 0].astype(int)
        ys = points[:, 1].astype(int)
        zs = points[:, 2].astype(int)

        input = np.zeros((1, self.voxel_size, self.voxel_size, self.voxel_size))
        input[0, xs, ys, zs] = 1

        return np.array(input), np.int(target)

    def __len__(self):
        return len(self.data)
