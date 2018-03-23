
import numpy as np
import scipy.io
import sys

def matfile2voxfile(mat_file, vox_file):
	mat = scipy.io.loadmat(mat_file)
	vox = mat2vox(mat)
	np.save(vox_file, vox)

def mat2vox(mat):
	vox = mat['instance']
	return vox

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'usage: python mat2vox.py path/to/input.mat path/to/output.npy'
		exit(0)

	mat_path = sys.argv[1]
	vox_path = sys.argv[2]
	matfile2voxfile(mat_path, vox_path)
	
	
