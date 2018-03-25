import os

import numpy as np
from torch.utils.data import Dataset


class ModelNet(Dataset):
    def __init__(self, data_path, split, resolution = 32):
        self.data_path = data_path
        self.split = split
        self.resolution = resolution

        if self.split in ['train', 'valid']:
            self.data = self.load_data(self.data_path + '.train')

            pivot = int(len(self.data) * .05)
            if self.split == 'train':
                self.data = self.data[:-pivot]
            else:
                self.data = self.data[-pivot:]
        elif self.split == 'test':
            self.data = self.load_data(self.data_path + '.test')
        else:
            raise AssertionError('unsupported data split "{0}"'.format(self.split))

    def load_data(self, path):
        data_path = os.path.dirname(path)

        data = []
        for line in open(path, 'r'):
            line = line.strip().split()
            if len(line) != 2:
                raise AssertionError('parsing error at {0} (line: {1})'.format(path, line))

            obj_path, syn_id = line
            data.append((os.path.join(data_path, obj_path), syn_id))
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
