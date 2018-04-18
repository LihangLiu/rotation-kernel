import os

import numpy as np
from torch.utils.data import Dataset

__all__ = ['ModelNet']


class ModelNet(Dataset):
    def __init__(self, data_path, split, voxel_size):
        self.data_path = data_path
        self.split = split
        self.voxel_size = voxel_size
        self.data = self.load_data(os.path.join(data_path, '{0}-{1}.txt'.format(split, voxel_size)))

    def load_data(self, data_path):
        data = []
        for line in open(data_path, 'r'):
            model_path, category = line.strip().split()
            data.append((os.path.join(self.data_path, model_path), np.int(category)))
        return data

    def __getitem__(self, index):
        model_path, target = self.data[index]
        data = np.load(model_path).astype(int)

        input = np.zeros((1, self.voxel_size, self.voxel_size, self.voxel_size))
        input[0, data[:, 0], data[:, 1], data[:, 2]] = 1
        return input, target

    def __len__(self):
        return len(self.data)
