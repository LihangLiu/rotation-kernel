from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ModelNet
from networks import ConvNet3D
from utils import set_cuda_devices
from utils.shell import mkdir
from utils.torch import ClassErrorMeter, Logger, load_snapshot, save_snapshot, to_var

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--gpu', default = '0')

    # dataset
    parser.add_argument('--data_path', default = '../data/')
    parser.add_argument('--voxel_size', default = 32, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 64, type = int)

    # network
    parser.add_argument('--kernel_mode', default = None, choices = ['3d', '2d+1d', '3d+rot', '2d+1d+rot'])

    # distillation
    parser.add_argument('--teacher', default = None)

    # training
    parser.add_argument('--epochs', default = 64, type = int)
    parser.add_argument('--snapshot', default = 1, type = int)
    parser.add_argument('--learning_rate', default = 0.0001, type = float)
    parser.add_argument('--lr_theta', default = 0.001, type = float)

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # cuda devices
    set_cuda_devices(args.gpu)

    # datasets & loaders
    data, loaders = {}, {}
    for split in ['train', 'valid', 'test']:
        data[split] = ModelNet(
            data_path = args.data_path,
            split = split,
            voxel_size = args.voxel_size
        )
        loaders[split] = DataLoader(
            dataset = data[split],
            batch_size = args.batch,
            shuffle = True,
            num_workers = args.workers
        )
    print('==> dataset loaded')
    print('[size] = {0} + {1} + {2}'.format(len(data['train']), len(data['valid']), len(data['test'])))

    # model
    model = ConvNet3D(
        channels = [1, 32, 64, 128, 256, 512],
        kernel_mode = args.kernel_mode,
        num_classes = 40,
    ).cuda()

    # teacher
    if args.teacher is not None:
        targs = load_snapshot(args.teacher, returns = 'args')

        teacher = ConvNet3D(
            channels = [1, 32, 64, 128, 256, 512],
            kernel_mode = targs.kernel_mode,
            num_classes = 40,
        ).cuda()

        load_snapshot(args.teacher, model = teacher)

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizers
    if 'rot' in args.kernel_mode:
        param_dict = dict(model.named_parameters())
        weight_params = [param_dict[k] for k in param_dict if 'theta' not in k]
        theta_params = [param_dict[k] for k in param_dict if 'theta' in k]
        optimizers = [
            torch.optim.Adam(weight_params, lr = args.learning_rate),
            torch.optim.Adam(theta_params, lr = args.lr_theta)
        ]
    else:
        optimizers = [
            torch.optim.Adam(model.parameters(), lr = args.learning_rate)
        ]

    # load snapshot
    if args.resume is not None:
        epoch = load_snapshot(args.resume, model = model, returns = 'epoch')
        print('==> snapshot "{0}" loaded'.format(args.resume))
    else:
        epoch = 0

    # save path
    save_path = os.path.join('..', 'exp', args.exp)
    mkdir(save_path, clean = False)

    # logger
    logger = Logger(save_path)

    # iterations
    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])
        print('==> epoch {0} (starting from step {1})'.format(epoch + 1, step + 1))

        # optimizer
        optimizer = optimizers[epoch % len(optimizers)]

        # training
        model.train()
        for inputs, targets in tqdm(loaders['train'], desc = 'train'):
            inputs = to_var(inputs)
            targets = to_var(targets, type = 'long')

            # forward
            optimizer.zero_grad()
            outputs = model.forward(inputs)

            if args.teacher is not None:
                results = teacher.forward(inputs)
                print(outputs.size(), results.size())
                print(outputs[0])
                print(results[0])

            # loss
            loss = criterion(outputs, targets)

            # logger
            logger.scalar_summary('train-loss', loss.data[0], step)
            step += targets.size(0)

            # backward
            loss.backward()
            optimizer.step()

        # testing
        model.eval()

        accuracy = {}
        for split in ['train', 'valid', 'test']:
            meter = ClassErrorMeter()

            for inputs, targets in tqdm(loaders[split], desc = split):
                inputs = to_var(inputs, volatile = True)
                targets = to_var(targets, type = 'long', volatile = True)

                # forward
                outputs = model.forward(inputs)
                meter.add(outputs, targets)

            accuracy[split] = meter.value()

        # logger
        for split in ['train', 'valid', 'test']:
            logger.scalar_summary('{0}-accuracy'.format(split), accuracy[split], step)

        # snapshot
        save_snapshot(
            path = os.path.join(save_path, 'latest.pth'),
            model = model,
            optimizer = optimizer,
            accuracy = accuracy,
            epoch = epoch + 1,
            args = args
        )

        if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
            save_snapshot(
                path = os.path.join(save_path, 'epoch-{0}.pth'.format(epoch + 1)),
                model = model,
                optimizer = optimizer,
                accuracy = accuracy,
                epoch = epoch + 1,
                args = args
            )
