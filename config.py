train_dataset_path = "./data/ModelNet_list/npy_list.30.points.train"
test_dataset_path = "./data/ModelNet_list/npy_list.30.points.test"
NUM_SYNS = 40


def get_folder_name():
    from os.path import realpath, dirname, basename
    return basename(dirname(realpath(__file__)))


version = get_folder_name()

ITER_MIN = 0
ITER_MAX = 51
save_interval = 10
test_interval = 1
lr_decay_interval = 100

batch_size = 32
base_lr = 1e-4
nf = 32

loss_csv = "./outputs/losses/loss_%s.csv" % (version)
param_prefix = "./outputs/params/param_%s_" % (version)
log_txt = "./outputs/log/log_%s.txt" % (version)
