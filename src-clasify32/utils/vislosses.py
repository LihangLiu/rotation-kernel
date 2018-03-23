import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
import time
import random



def pltLoss(vec_loss, label):
    plt.plot(vec_loss, label=label)

def parseLossFile(txt_path):
    vec_loss = []
    with open(txt_path, 'r') as f:
        for line in f:
            segs = line.strip().split()
            vec_loss.append(float(segs[-1]))
            # vec_loss.append(float(segs[-3]))
    return np.array(vec_loss)

def smooth_loss(vec_loss, step=4):
    res = []
    n = vec_loss.shape[0]
    for i in range(n):
        if i + 1 - step >= 0:
            average_loss = np.mean(vec_loss[i+1-step: i+1])
        else:
            average_loss = np.mean(vec_loss[0:i+1])
        res.append(average_loss)
    return np.array(res)

if __name__ == '__main__':
    txt_list = [        
                '../../outputs/losses/loss_shapenets32-000.csv',                
                '../../outputs/losses/loss_shapenets32-001.csv',                
                '../../outputs/losses/loss_shapenets32-002.csv',                
                ]
    legend_list = (
    				'0',
                    '1',
                    '2',
                    '3',
                    '4',
                    '5',
                    '6',
                    '7',
                    '8',
                    '9',
                    '10',
                    # 'Traditional',
                    # 'XY-Z,XZ-Y,YZ-X',
                    # 'Prefixed Rotation theta',
                    # 'Trainable theta',
                    # 'Prefixed theta + RI',
                )

    for i, txt_path in enumerate(txt_list):
        this_dir = osp.dirname(__file__)
        vec_loss = parseLossFile(osp.join(this_dir, txt_path))
        # if i > 2:
        #     vec_loss = vec_loss[::2]
        vec_loss = smooth_loss(vec_loss)
        print txt_path, vec_loss.shape, np.max(vec_loss)
        pltLoss(vec_loss, legend_list[i])
    plt.title('test accuracy')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()








