#!/bin/bash
source activate snakes
python train.py --train_path="../../data/ModelNet_list/npy_list_3.64.points.train" --valid_path="../../data/ModelNet_list/npy_list_3.64.points.fakevalid" --test_path="../../data/ModelNet_list/npy_list_3.64.points.test" --num_syns="40" --voxel_size="64" --workers="8" --batch_size="32" --max_iter="51" --save_interval="1" --lr_base="0.0001" --lr_theta="0.001" --lr_step_size="6" --optimizer="adam" --alter_step_size="1" --nf="12" --kernel_mode="3d_mn_init1" --num_theta="2" --loss_csv="../../outputs/losses/loss_modelnet64_3d_mn_init1_nf12_nt2.csv" --param_prefix="../../outputs/params/param_modelnet64_3d_mn_init1_nf12_nt2_"
