from __future__ import print_function

import argparse
import os

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnx.torch as tx
from data import ModelNet
from models import ConvRotateNet3d
from utilx import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--gpu', default = '0')

    parser.add_argument('--data_path', default = './data/')
    parser.add_argument('--voxel_size', default = 32, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 64, type = int)

    parser.add_argument('--kernel_mode', choices = ['3d', '2d+1d', '1d+1d+1d'])
    parser.add_argument('--kernel_rotate', action = 'store_true')

    parser.add_argument('--epochs', default = 64, type = int)
    parser.add_argument('--snapshot', default = 1, type = int)
    parser.add_argument('--learning_rate', default = 1e-4, type = float)
    parser.add_argument('--weight_decay', default = 1e-3, type = float)
    parser.add_argument('--step_size', default = 8, type = int)
    parser.add_argument('--gamma', default = 4e-1, type = float)

    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    args.gpu = set_cuda_visible_devices(args.gpu)

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

    model = ConvRotateNet3d(
        channels = [1, 32, 64, 128, 256, 512],
        features = [512, 128, 40],
        kernel_mode = args.kernel_mode,
        kernel_rotate = args.kernel_rotate
    )

    if 'rot' in args.kernel_mode:
        # fixme: clean up
        param_dict = dict(model.named_parameters())
        weight_params = [param_dict[k] for k in param_dict if 'theta' not in k]
        theta_params = [param_dict[k] for k in param_dict if 'theta' in k]
        optimizers = [
            torch.optim.Adam(weight_params, lr = args.learning_rate, weight_decay = args.weight_decay),
            torch.optim.Adam(theta_params, lr = args.learning_rate)
        ]
    else:
        optimizers = [
            torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
        ]

    if args.resume is not None:
        # fixme: load optimizer
        epoch = tx.load_snapshot(args.resume, model = model, returns = 'epoch')
        print('==> snapshot "{0}" loaded'.format(args.resume))
    else:
        epoch = 0

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    save_path = os.path.join('exp', args.exp)
    mkdir(save_path, clean = args.resume is None)

    logger = tx.Logger(save_path)

    schedulers = []
    for optimizer in optimizers:
        # fixme: last epoch
        schedulers.append(torch.optim.lr_scheduler.StepLR(
            optimizer = optimizer,
            step_size = args.step_size,
            gamma = args.gamma,
        ))

    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])
        print('==> epoch {0} (starting from step {1})'.format(epoch + 1, step + 1))

        schedulers[epoch % len(schedulers)].step()
        optimizer = optimizers[epoch % len(optimizers)]

        model.train()
        for inputs, targets in tqdm(loaders['train'], desc = 'train'):
            inputs = tx.as_variable(inputs).float()
            targets = tx.as_variable(targets).long()

            optimizer.zero_grad()
            outputs = model.forward(inputs)

            loss = cross_entropy(outputs, targets)

            logger.scalar_summary('train-loss', loss.item(), step)
            step += targets.size(0)

            loss.backward()
            optimizer.step()

        model.eval()

        accuracy = {}
        for split in ['train', 'valid', 'test']:
            meter = tx.meters.ClassErrorMeter()

            for inputs, targets in tqdm(loaders[split], desc = split):
                inputs = tx.mark_volatile(inputs).float()
                targets = tx.mark_volatile(targets).long()

                outputs = model.forward(inputs)
                meter.add(outputs, targets)

            accuracy[split] = meter.value()

        if (epoch + 1) % len(optimizers) == 0:
            for split in ['train', 'valid', 'test']:
                logger.scalar_summary('{0}-accuracy'.format(split), accuracy[split], step)

        tx.save_snapshot(
            path = os.path.join(save_path, 'latest.pth'),
            model = model,
            optimizer = optimizer,
            accuracy = accuracy,
            epoch = epoch + 1,
            args = args
        )

        if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:
            tx.save_snapshot(
                path = os.path.join(save_path, 'epoch-{0}.pth'.format(epoch + 1)),
                model = model,
                optimizer = optimizer,
                accuracy = accuracy,
                epoch = epoch + 1,
                args = args
            )
