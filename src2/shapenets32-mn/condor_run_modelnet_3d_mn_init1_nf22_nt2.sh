#!/bin/bash
source activate snakes
python train.py --train_path="../../data/train-32.txt" --valid_path="../../data/valid-32.txt" --test_path="../../data/test-32.txt" --num_syns="40" --voxel_size="32" --workers="8" --batch_size="64" --max_iter="51" --save_interval="10" --lr_base="0.0001" --lr_theta="0.001" --lr_step_size="10" --optimizer="adam" --alter_step_size="1" --nf="22" --kernel_mode="3d_mn_init1" --num_theta="2" --loss_csv="../../outputs/losses/loss_modelnet_3d_mn_init1_nf22_nt2.csv" --param_prefix="../../outputs/params/param_modelnet_3d_mn_init1_nf22_nt2_"
