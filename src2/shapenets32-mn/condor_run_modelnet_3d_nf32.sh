#!/bin/bash
source activate snakes
python train.py --train_path="../../data/train-32.txt" --valid_path="../../data/valid-32.txt" --test_path="../../data/test-32.txt" --num_syns="40" --voxel_size="32" --workers="8" --batch_size="128" --max_iter="81" --save_interval="10" --lr_base="0.0001" --lr_theta="0.001" --lr_step_size="20" --optimizer="adam" --alter_step_size="1" --nf="32" --kernel_mode="3d" --num_theta="1" --loss_csv="../../outputs/losses/loss_modelnet_3d_nf32.csv" --param_prefix="../../outputs/params/param_modelnet_3d_nf32_"
