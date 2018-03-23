
import numpy as np
import scipy.io
import os
import sys
import glob

from mat2points import matfile2pointsfile


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print 'usage: python mat2vox.py npy_list.txt'
		exit(0)

	npy_list_path = sys.argv[1]

	with open(npy_list_path+'.train', 'w') as train_f:
		with open(npy_list_path+'.test', 'w') as test_f:
			for line in open(npy_list_path, 'r'):
				if 'train' in line:
					train_f.write(line)
				if 'test' in line:
					test_f.write(line)
	print('created', npy_list_path+'.train')
	print('created', npy_list_path+'.test')
		


