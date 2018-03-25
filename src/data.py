import os

import numpy as np
from torch.utils.data import Dataset


class ModelNet(Dataset):
    def __init__(self, data_path, split, resolution = 32):
        self.data_path = data_path
        self.split = split
        self.resolution = resolution

        self.data = self.load_data(os.path.join(self.data_path, '{0}-{1}.txt'.format(self.split, self.resolution)))

    def load_data(self, path):
        data = []
        for line in open(path, 'r'):
            a, b = line.strip().split()
            data.append((os.path.join(self.data_path, a), b))
        return data

    def __getitem__(self, index):
        path, target = self.data[index]

        points = np.load(path)
        xs = points[:, 0].astype(int)
        ys = points[:, 1].astype(int)
        zs = points[:, 2].astype(int)

        input = np.zeros((1, self.resolution, self.resolution, self.resolution))
        input[0, xs, ys, zs] = 1

        return np.array(input), np.int(target)

    def __len__(self):
        return len(self.data)
