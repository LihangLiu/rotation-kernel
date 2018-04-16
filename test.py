from __future__ import print_function

import argparse
import os

from utilx.cli import *
from learnx.torch.io.snapshot import load_snapshot

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # save path
    save_path = os.path.join( 'exp', args.exp)

    # test accuracy
    valid, test = 0, 0
    for snapshot_path in ls(save_path, suffix = '.pth'):
        path = os.path.join(save_path, snapshot_path)
        accuracy = load_snapshot(path, returns = 'accuracy')

        if accuracy['valid'] > valid:
            valid, test = accuracy['valid'], accuracy['test']
    print('==> test accuracy = {0:.2f}%'.format(test))
