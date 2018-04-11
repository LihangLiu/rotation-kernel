import os

import _init_paths
from utils.condorhelper import *
from utils.filehelper import *

# specify arguments
def get_args(exp_id):
    args = {}
    args['train_path'] = '../../data/train-32.txt'
    args['valid_path'] = '../../data/valid-32.txt'
    args['test_path'] = '../../data/test-32.txt'
    args['num_syns'] = 40
    args['voxel_size'] = 32

    args['workers'] = 8
    args['batch_size'] = 128
    args['max_iter'] = 61
    args['save_interval'] = 10

    args['lr_base'] = 1e-4
    args['lr_theta'] = 1e-3
    args['lr_step_size'] = 40
    args['optimizer'] = 'adam'

    args['nf'] = 32
    args['kernel_mode'] = '3d_rot_multheta'
    args['num_theta'] = 4

    args['loss_csv'] = "../../outputs/losses/loss_%s.csv"%(exp_id)
    args['param_prefix'] = "../../outputs/params/param_%s_"%(exp_id)

    return args

# specify the experiment id
exp_id = "3d_rot_multheta"


if __name__ == '__main__':
    ##############
    ### don't edit
    ##############
    print(exp_id)
    shell_file = 'condor_run_%s.sh' % (exp_id)
    py_script = 'train.py'
    py_args = get_args(exp_id)
    condor_file = 'condor_submit_%s' % (exp_id)
    condor_log_path = "../../outputs/log/log_%s_" % (exp_id)
    env = 'snakes'

    # generate executable file
    exec_content = exec_file_content(py_script, py_args, env=env)
    write_2_file(exec_content, shell_file, mode='w')

    # generate condor config file
    condor_content = condor_config_content(
        project_desc = 'rotation kernel', 
        condor_log_path = condor_log_path,
        exec_file_path = shell_file
    )
    write_2_file(condor_content, condor_file, mode='w')

    # submit job
    os.system('chmod 755 %s'%(shell_file))
    os.system("condor_submit %s" % (condor_file))

    