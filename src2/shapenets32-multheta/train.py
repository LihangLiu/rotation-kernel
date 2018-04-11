import argparse
import time
import numpy as np
import sys
from tqdm import tqdm

from os.path import exists
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import _init_paths
from utils.dataset import ModelNet
from utils.torchhelper import *
from utils.filehelper import *
from model import ConvNet


def train(net, optimizer, no_epoch, loader, spar_factor=0):
    net.train()
    loss_list = []
    for inputs, targets in tqdm(loader, desc = 'train'):
        inputs = to_var(inputs)
        targets = to_var(targets, type = 'long')

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if spar_factor != 0:
            loss += net.get_spar_regular_loss()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.data[0])

    return np.mean(loss_list)

def test(net, no_epoch, loader, meter_class):
    net.eval()
    meter = meter_class()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc = 'eval'):
            inputs = to_var(inputs)
            targets = to_var(targets, type = 'long')

            outputs = net(inputs)
            meter.add(outputs, targets) 
    return meter

if __name__ == '__main__':
    ### print args
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')

    # dataset
    parser.add_argument('--train_path', default = '../../data/train-32.txt')
    parser.add_argument('--valid_path', default = '../../data/valid-32.txt')
    parser.add_argument('--test_path', default = '../../data/test-32.txt')
    parser.add_argument('--voxel_size', default = 32, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch_size', default = 32, type = int)

    # network
    parser.add_argument(
        '--kernel_mode', 
        default = '3d', 
        choices = ['3d', '3d_rot_multheta']
    )
    parser.add_argument('--num_theta', default = 1, type = int)
    parser.add_argument('--nf', default = 24, type = int)
    parser.add_argument('--num_syns', default = 40, type = int)

    # training
    parser.add_argument('--max_iter', default = 61, type = int)
    parser.add_argument('--save_interval', default = 10, type = int)
    parser.add_argument('--lr_base', default = 1e-4, type = float)
    parser.add_argument('--lr_theta', default = 1e-3, type = float)
    parser.add_argument('--lr_step_size', default = 20, type = int)
    parser.add_argument('--optimizer', default = 'adam', type = str)

    # logging
    parser.add_argument('--loss_csv', default = "loss.csv")
    parser.add_argument('--param_prefix', default = None)

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    ############
    # load data
    ############
    loaders = {}
    datas = {
        'train': ModelNet(data_path = args.train_path, voxel_size = args.voxel_size),
        'valid': ModelNet(data_path = args.valid_path, voxel_size = args.voxel_size),
        'test': ModelNet(data_path = args.test_path, voxel_size = args.voxel_size)
    }
    for split in ['train', 'valid', 'test']:
        data = datas[split]
        loaders[split] = DataLoader(
            data, 
            batch_size = args.batch_size, 
            shuffle = True, 
            num_workers = args.workers
        )
        print('{0} data size: {1}'.format(split, len(data)))

    #############
    # build net
    #############
    net = ConvNet(
        nf = args.nf, 
        num_syns = args.num_syns, 
        kernel_mode = args.kernel_mode,
        num_theta = args.num_theta,
        input_size = args.voxel_size
    ).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    if args.optimizer == 'adam':
        opt_class = optim.Adam
        opt_args = {}
    elif args.optimizer == 'sgd':
        opt_class = optim.SGD
        opt_args = {'momentum': 0.9}

    param_dict = dict(net.named_parameters())
    if 'rot' in args.kernel_mode:
        weight_params = filter_dict(lambda key: 'theta' not in key, param_dict).values()
        theta_params = filter_dict(lambda key: 'theta' in key, param_dict).values()
        optimizer = AlterOptim(
            optim0 = opt_class(weight_params, lr=args.lr_base, **opt_args), 
            optim1 = opt_class(theta_params, lr=args.lr_theta, **opt_args), 
            logging = True
        )
    else:
        optimizer = opt_class(param_dict.values(), lr=args.lr_base, **opt_args)

    print(net)
    print(param_dict.keys())

    #############
    # train
    #############
    # load snapshot
    if not args.param_prefix is None:
        start_iter = load_snapshot(net, args.param_prefix)
        if start_iter:
            print('==> Load model from {0}{1}'.format(args.param_prefix, start_iter-1))
    else:
        start_iter = 0
        net.apply(weights_init)
        print('==> Random init')

    # lr scheduler
    if 'rot' in args.kernel_mode:
        lr_schedulers = [
            StepLR(optimizer.optim0, step_size=args.lr_step_size, gamma=0.1, last_epoch=-1),
            StepLR(optimizer.optim1, step_size=args.lr_step_size, gamma=0.1, last_epoch=-1),
        ]
        lr_schedulers[0].step(start_iter-1)
        lr_schedulers[1].step(start_iter-1)
    else:
        lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1, last_epoch=-1)
        lr_scheduler.step(start_iter-1)

    # start training
    for epoch in range(start_iter, args.max_iter): 
        start = time.time()
        print('[Epoch {0}]'.format(epoch))

        # learning rate decay
        if 'rot' in args.kernel_mode:
            lr_schedulers[0].step()
            lr_schedulers[1].step()
            optimizer.print_lr()
        else:
            lr_scheduler.step()
            print('lr: {}'.format([g['lr'] for g in optimizer.param_groups]))

        # train
        if 'rot' in args.kernel_mode:
            optimizer.set_optim(epoch % 2)
        
        train_loss = train(net, optimizer, epoch, loaders['train'])

        # test
        accuracy = {}
        for split in ['train', 'valid', 'test']:
            meter = test(net, epoch, loaders[split], meter_class=ErrorMeter)
            accuracy[split] = meter.value()

        # output losses
        msg = "{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(
            epoch, train_loss, accuracy['train'], accuracy['valid'], accuracy['test']
        )
        write_2_file(msg, args.loss_csv, new_line=True)
        print(msg)
        print('Time used: {0:.2f} s'.format(time.time() - start))

        # save snapshot
        if epoch%args.save_interval == 0:
            save_snapshot(net, args.param_prefix, epoch)
            print('==> Save model to {0}{1}'.format(args.param_prefix, epoch))
            






