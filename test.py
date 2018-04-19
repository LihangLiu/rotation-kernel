from __future__ import print_function

import argparse
import os

from learnx.torch.io import *
from utilx.cli import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', default = 'default')

    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    save_path = os.path.join('exp', args.exp)

    valid, test = 0, 0
    for snapshot_path in ls(save_path, contains = '.pth'):
        path = os.path.join(save_path, snapshot_path)
        accuracy = load_snapshot(path, returns = 'accuracy')

        if accuracy['valid'] > valid:
            valid, test = accuracy['valid'], accuracy['test']
    print('==> test accuracy = {0:.2f}%'.format(test))
