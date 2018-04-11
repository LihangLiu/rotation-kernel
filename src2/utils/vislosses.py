import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import dirname, basename, join
import sys
import time
import random

def pltLoss(vec_loss, label):
    plt.plot(vec_loss, label=label)

def parseLossFile(txt_path, pos=-1):
    vec_loss = []
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            segs = line.strip().split()
            vec_loss.append(float(segs[pos]))
    return np.array(vec_loss)

def get_legend(txt_path):
    name = basename(txt_path)
    if name.startswith('loss_'):
        name = name[5:]
    if name.endswith('.csv'):
        name = name[:-4]
    return name

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def get_max_by_val(test_loss, val_loss):
    index = np.argmax(val_loss)
    return test_loss[index]

if __name__ == '__main__':
    txt_list = [        
        # ('../../outputs/losses/loss_shapenets32-000.csv', ''),
        # ('../../outputs/losses/loss_shapenets32-000_nf24.csv', ''),
        # ('../../outputs/losses/loss_shapenets32-001.csv', ''),
        # ('../../outputs/losses/loss_shapenets32-001_nf24.csv', ''),
        # ('../../outputs/losses/loss_shapenets32-002.csv', ''),
        # ('../../outputs/losses/loss_shapenets32-002.1.csv', ''),

        # nf = 32 and lr decay
        # ('../../outputs/losses/loss_3d.csv', ''),
        # ('../../outputs/losses/loss_2d_1d.csv', ''),
        # ('../../outputs/losses/loss_decomp_reg0001_plain.csv', ''),

        # # ('../../outputs/losses/loss_3d_rot_fast_lrdecay.csv', 'alter'),
        # ('../../outputs/losses/loss_3d_rot_fast_lrdecay_repeat.csv', 'alter'),
        # # ('../../outputs/losses/loss_3d_rot_lr01.csv', 'alter'),
        # ('../../outputs/losses/loss_2d_1d_rot_fast_lrdecay.csv', 'alter'),
        # # ('../../outputs/losses/loss_decomp_reg0001.csv', 'alter'),

        # no lr decay
        # ('../../outputs/losses/loss_3d_rot.csv', 'alter'),
        # ('../../outputs/losses/loss_3d_rot_tanh.csv', 'alter'),
        # ('../../outputs/losses/loss_3d_rot_0init.csv', 'alter'),
        # ('../../outputs/losses/loss_3d_rot_spar01.csv', 'alter'),
        # ('../../outputs/losses/loss_2d_1d_rot.csv', 'alter'),

        # sgd
        # ('../../outputs/losses/loss_3d.csv', 'alter'),
        # ('../../outputs/losses/loss_3d_rot.csv', 'alter'),
        # ('../../outputs/losses/loss_3d_sgd.csv', ''),
        # ('../../outputs/losses/loss_3d_rot_sgd.csv', 'alter'),

        # lrdecay
        # ('../../outputs/losses/loss_3d_lrdecay.csv', ''),
        # ('../../outputs/losses/loss_3d_rot_lrdecay.csv', ''),
        

        # multi theta
        # ('../../outputs/losses/loss_vox32_3d_rot_ntheta1_nf8.csv', 'alter'),
        # ('../../outputs/losses/loss_vox32_3d_rot_max_ntheta4_nf8.csv', 'alter'),
        # ('../../outputs/losses/loss_vox32_3d_rot_max_ntheta6_nf8.csv', 'alter'),
        # ('../../outputs/losses/loss_vox32_3d_rot_max_ntheta4_nf8_run2.csv', 'alter'),
        # ('../../outputs/losses/loss_vox32_3d_rot_max_rtheta_ntheta4_nf8.csv', 'alter'),

        # ('../../outputs/losses/loss_3d_rot.csv', 'alter'),
        # ('../../outputs/losses/loss_vox32_3d_rot_max_ntheta4_nf32.csv', 'alter'),
        # ('../../outputs/losses/loss_vox32_3d_rot_max_ntheta6_nf32.csv', 'alter'),

        # vox size = 64
        ('../../outputs/losses/loss_vox64_3d.csv', ''),
        ('../../outputs/losses/loss_vox64_3d_rot.csv', 'alter'),
        # ('../../outputs/losses/loss_vox64_3d_rot_tanh.csv', 'alter'),
        ('../../outputs/losses/loss_vox64_3d_rot_max_ntheta4.csv', 'alter'),
        ('../../outputs/losses/loss_vox64_2d_1d.csv', ''),
        ('../../outputs/losses/loss_vox64_2d_1d_rot.csv', 'alter'),

    ]

    for i, (txt_path, option) in enumerate(txt_list):
        this_dir = dirname(__file__)
        legend = get_legend(txt_path)

        # parse
        test_loss = parseLossFile(join(this_dir, txt_path), pos=-1)
        val_loss = parseLossFile(join(this_dir, txt_path), pos=-2)

        if option == 'alter':
            test_loss = test_loss[1::2]
            val_loss = val_loss[1::2]

        test_loss = test_loss[:30]
        val_loss = val_loss[:30]

        # smooth
        test_loss = smooth(test_loss, 0.6)
        val_loss = smooth(val_loss, 0.6)

        print('[{0}] len:{1} max:{2:.4f} max by val:{3:.4f}'.format(
            legend, len(test_loss), np.max(test_loss), get_max_by_val(test_loss, val_loss)
        ))
        pltLoss(test_loss, legend)
    plt.title('test accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()








