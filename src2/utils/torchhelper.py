import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from os.path import exists
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0.0, 0.02)

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)

    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def filter_false_requires_grad(param_list):
    return filter(lambda p: p.requires_grad, param_list)

def filter_dict(func, input_dict):
    new_dict = {}
    for key in input_dict:
        if func(key):
            new_dict[key] = input_dict[key]
    return new_dict

def fetch_lastest_param_path(param_prefix, start_i, end_i):
    for i in range(end_i, start_i-1, -1):
        param_path = param_prefix + str(i)
        if exists(param_path):
            return i, param_path
    return -1, None

def load_snapshot(net, param_prefix, max_no_epoch=10000):
    """
    return no_next_epoch
    """
    saved_iter, saved_param_path = fetch_lastest_param_path(param_prefix, 0, max_no_epoch)
    if saved_param_path:
        net.load_state_dict(torch.load(saved_param_path))
        return saved_iter + 1
    else:
        return None

def save_snapshot(net, param_prefix, no_epoch):
    path = '{0}{1}'.format(param_prefix, no_epoch)
    torch.save(net.state_dict(), path)

def torchmax(tlist):
    n = len(tlist)
    res = tlist[0]
    if n == 1:
        return res
    for i in range(1, n):
        res = torch.max(res, tlist[i])
    return res

def to_var(tensor, type='float'):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if type == 'float':
        return Variable(tensor.cuda().float())
    elif type == 'long':
        return Variable(tensor.cuda().long())    
    else:
        raise NotImplementedError('type: {0} not implemented'.format(p))

def to_tensor(tensor, type='float'):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if type == 'float':
        return tensor.cuda().float()
    elif type == 'long':
        return tensor.cuda().long()
    else:
        raise NotImplementedError('type: {0} not implemented'.format(p))

def to_np(inputs, type = 'float'):
    if isinstance(inputs, Variable):
        inputs = inputs.data

    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()

    inputs = np.array(inputs, dtype = type)
    return inputs

def cal_table(gt_labels, pred_labels, table_size):
    """
    gt_labels: (n,)
    pred_labels: (n,)
    """
    table = np.zeros([table_size, table_size])
    for g, p in zip(gt_labels, pred_labels):
        table[g, p] += 1
    return table

class ErrorMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt_correct = 0.0
        self.cnt_total = 0.0

    def add(self, outputs, targets):
        _, prediction = torch.max(outputs, 1)
        c = (prediction == targets)
        self.cnt_correct += float(torch.sum(c.data))
        self.cnt_total += float(c.numel())

    def value(self):
        return 0 if self.cnt_total == 0 else self.cnt_correct / self.cnt_total

class AlterOptim:
    def __init__(self, optim0, optim1, logging=False):
        self.optim0 = optim0
        self.optim1 = optim1
        self.cur_optim = None
        self.logging = logging

    def set_optim(self, index=0):
        if index == 0:
            self.cur_optim = self.optim0
            if self.logging: print('=> Set Optim to: 0')
        elif index == 1:
            self.cur_optim = self.optim1
            if self.logging: print('=> Set Optim to: 1')
        else:
            raise NotImplementedError('Index {0} exceed the limit'.format(index))

    def zero_grad(self):
        self.cur_optim.zero_grad()

    def step(self):
        self.cur_optim.step()

    def print_lr(self):
        s = 'lr: {} {}'.format(
            [g['lr'] for g in self.optim0.param_groups],
            [g['lr'] for g in self.optim1.param_groups],
        )
        print(s)



