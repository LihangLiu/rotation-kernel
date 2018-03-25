import numpy as np
from os.path import dirname, basename, join
import os
import sys
import time
import random
import scipy.io
from multiprocessing import Pool, cpu_count

class Dataset:

    def __init__(self, dataset_path, max_num=None, if_shuffle=True, seed=None, hard_datapass=False, multiprocessing=True):
        self.dataset_path = dataset_path
        self.if_shuffle = if_shuffle
        self.hard_datapass = hard_datapass
        self.multiprocessing = multiprocessing
        print("Datapath:", self.dataset_path)

        self.index_in_epoch = 0
        if max_num is None:
            self.examples = np.array(self.read_txt(dataset_path))
        else:
            self.examples = np.array(self.read_txt(dataset_path))[:max_num]
        self.num_examples = len(self.examples)
        print('Dataset size: ', self.num_examples)

        if seed is None:
            print('random sample enabled')
        else:
            np.random.seed(seed)
            print('generate random sample by seed', seed)
        if self.if_shuffle: 
            np.random.shuffle(self.examples)

        if self.multiprocessing:
            # print('multi processing forced to be stopped')
            # self.multiprocessing = False
            cpu_cnt = cpu_count()
            print('Cpu used: 2')
            self.pool = Pool(2)

    def read_txt(self, txtFile):
        """
        return [(obj_path,syn_id=-1)]
        """
        txtDir = dirname(txtFile)
        data_pair_list = []
        for line in open(txtFile, 'r'):     # obj_path syn_id
            line = line.strip().split()     # -> (obj_path,syn_id=-1)
            if len(line) != 2:
                print('error in parse', txtFile, line)
                exit(1)
            obj_path, syn_id = line
            data_pair_list.append((join(txtDir, obj_path), syn_id))

        return data_pair_list

    def num_valid_batch(self, batch_size):
        num_batch = int(self.num_examples / batch_size)
        if num_batch * batch_size != self.num_examples and self.hard_datapass:
            return num_batch + 1
        return num_batch

    def next_batch(self, batch_size):
        """
        hard_datapass: if reach the end, return the remaining examples only
        """
        assert batch_size <= self.num_examples

        if self.index_in_epoch >= self.num_examples:
            self.index_in_epoch = 0

        if self.index_in_epoch + batch_size > self.num_examples:
            if self.hard_datapass:
                batch = self.read_data(self.index_in_epoch, self.num_examples)
                if self.if_shuffle: np.random.shuffle(self.examples)
                self.index_in_epoch = 0
            else:
                if self.if_shuffle: np.random.shuffle(self.examples)
                batch = self.read_data(0, batch_size)
                self.index_in_epoch = batch_size
        else:
             batch = self.read_data(self.index_in_epoch, self.index_in_epoch + batch_size)
             self.index_in_epoch += batch_size

        return batch

    SUFFIX_64_POINTS = '64.points.npy'
    SUFFIX_30_POINTS = '30.points.npy'

    def read_data(self, start, end):
        """
        vox: (batch_size, 1, n,n,n)
        syn_id: (batch_size,)
        """
        batch = {'vox':[], 'syn_id':[]}
        batch['syn_id'] = np.array([int(syn_id) for fname, syn_id in self.examples[start:end]])
        fname_list = [fname for fname, syn_id in self.examples[start:end]]
        if fname_list[0].endswith(self.SUFFIX_64_POINTS):
            load_handler = load_points_2_vox64
        elif fname_list[0].endswith(self.SUFFIX_30_POINTS):
            load_handler = load_points_2_vox32
        else:
            print('unsupported format', fname, syn_id)
            exit(0)

        if self.multiprocessing:     # use multiprocessing for speedup
            batch['vox'] = np.array(self.pool.map(load_handler, fname_list))
        else:
            batch['vox'] = np.array([load_handler(fname) for fname in fname_list])

        return batch

def load_points_2_vox64(fname):
    points = np.load(fname)
    vox = points2vox(points,64)
    return vox

def load_points_2_vox32(fname):
    points = np.load(fname)
    vox = points2vox(points,32)
    return vox

def points2vox(points,N):
    """ points: (n,3)
        (n,0) -> x
        (n,1) -> y
        (n,2) -> z
        vox: (1,N,N,N)
    """
    xs = points[:,0].astype(int)
    ys = points[:,1].astype(int)
    zs = points[:,2].astype(int)
    vox = np.zeros((1,N,N,N))
    vox[0,xs,ys,zs] = 1
    return vox


if __name__ == '__main__':
    # npylist_path = "ModelNet_list/npy_list_3.64.points.test"
    npylist_path = "ModelNet_list/npy_list.30.points.test"
    dataset = Dataset(npylist_path, if_shuffle=True, hard_datapass=False, multiprocessing=True)
    print(dataset.num_valid_batch(32))
    for i in range(4):
        s = time.time()
        data_dict = dataset.next_batch(32)
        print(data_dict['vox'].shape)
        print(data_dict['syn_id'].shape)
        # print(data_dict['syn_id'])
        print('time used', time.time() - s)








