import argparse
import time
from os.path import exists
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import sys

import _init_paths
import dataset_clasify as dataset
from networks import ConvNet3D, weights_init
import config as myconfig

def fetch_lastest_param_path(param_prefix, start_i, end_i):
    for i in range(end_i, start_i-1, -1):
        param_path = param_prefix + str(i)
        if exists(param_path):
            return i, param_path
    return -1, None

def evaluate(mat_labels_, labels):
    """
    mat_labels_: (batch_size, num_syns)
    labels: (batch_size,)
    """
    _, labels_ = torch.max(mat_labels_, 1)
    c = (labels_ == labels)
    cnt_correct = torch.sum(c.data)
    cnt_total = c.numel()
    return cnt_correct, cnt_total

def CudaFloatVariable(tensor, volatile=False):
    return Variable(tensor.cuda().float(), volatile=volatile)

def CudaLongVariable(tensor, volatile=False):
    return Variable(tensor.cuda().long(), volatile=volatile)

if __name__ == '__main__':

    ############
    # train data
    ############
    train_data = dataset.Dataset(myconfig.train_dataset_path, hard_datapass=False, multiprocessing=True)
    test_data = dataset.Dataset(myconfig.test_dataset_path, hard_datapass=True, multiprocessing=True)
    # train_data = dataset.Dataset(myconfig.train_dataset_path, max_num=100, hard_datapass=False, multiprocessing=True)
    # test_data = dataset.Dataset(myconfig.test_dataset_path, max_num=100, hard_datapass=True, multiprocessing=True)

    #############
    # parameters
    #############
    batch_size = myconfig.batch_size
    learning_rate = myconfig.base_lr
    nf = myconfig.nf

    #############
    # build net
    #############
    net = ConvNet3D(channels =nf, num_classes =myconfig.NUM_SYNS)
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)

    ############
    # print net
    ############
    print(net, flush=True)

    #############
    # train
    #############
    ### look for existing model
    saved_iter, saved_param_path = fetch_lastest_param_path(myconfig.param_prefix, myconfig.ITER_MIN, myconfig.ITER_MAX)
    if saved_param_path:
        start_iter = saved_iter
        net.load_state_dict(torch.load(saved_param_path))
        print('loaded param:', saved_param_path)
    else:
        start_iter = myconfig.ITER_MIN
        net.apply(weights_init)
        print('no param found, random init')

    ### start training
    for epoch in range(start_iter, myconfig.ITER_MAX): 
        ### save checkpoints
        if epoch%myconfig.save_interval == 0:
            path = myconfig.param_prefix + str(epoch)
            torch.save(net.state_dict(), path)
            print('saved param to', path, flush=True)

        start = time.time()
        loss_list = []
        ### train on one epoch ### 
        net.train()     # turn on train mode
        train_cnt_correct, train_cnt_total = 0.0, 0.0000001
        for i in range(train_data.num_valid_batch(batch_size)): 
            s = time.time()
            data_dict = train_data.next_batch(batch_size)
            vox, labels = data_dict['vox'], data_dict['syn_id']
            vox = CudaFloatVariable(torch.from_numpy(vox))
            labels = CudaLongVariable(torch.from_numpy(labels))

            optimizer.zero_grad()
            mat_labels_ = net(vox)
            loss = criterion(mat_labels_, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.data[0])

            cnt_correct, cnt_total = evaluate(mat_labels_, labels)
            train_cnt_correct += cnt_correct
            train_cnt_total += cnt_total
            if i%1000 == 0:
                print('time used', time.time() - s)
                print(epoch, i, "loss:", loss.data[0], "train:", train_cnt_correct/train_cnt_total, flush=True)

        ### test on one epoch ### 
        net.eval()     # turn on test mode
        test_cnt_correct, test_cnt_total = 0.0, 0.0000001
        if epoch%myconfig.test_interval == 0:
            for i in range(test_data.num_valid_batch(batch_size)):
                data_dict = test_data.next_batch(batch_size)
                vox, labels = data_dict['vox'], data_dict['syn_id']
                vox = CudaFloatVariable(torch.from_numpy(vox), volatile=True)     # !!! volatile for purely inference mode
                labels = CudaLongVariable(torch.from_numpy(labels), volatile=True)

                mat_labels_ = net(vox)

                cnt_correct, cnt_total = evaluate(mat_labels_, labels)
                test_cnt_correct += cnt_correct
                test_cnt_total += cnt_total
                if i%1000 == 0:
                    print(epoch, i, "test", test_cnt_correct/test_cnt_total, flush=True)

        ### output losses ###
        with open(myconfig.loss_csv, 'a') as f:
            loss_list = torch.FloatTensor(loss_list)
            msg = "%d %.6f %.6f %.6f"%(epoch, torch.mean(loss_list), train_cnt_correct/train_cnt_total, test_cnt_correct/test_cnt_total)
            f.write(msg + '\n')
            print(msg, flush=True)
            print('time used %.2f'%(time.time()-start), flush=True)
            






