
import numpy as np
import scipy.io
import sys

def vox2points(vox):
## vox: (w,d,h)
## xs: (n,1)
## rbgs: (n,1)
	xs,ys,zs = np.nonzero(vox)
	xs,ys,zs = [np.expand_dims(vs,axis=1) for vs in [xs,ys,zs]]
	points = np.concatenate((xs,ys,zs),axis=1)
	return points

def mat2points(mat):
	vox = mat['binary']
	points = vox2points(vox)
	return points

def matfile2pointsfile(mat_file, points_file):
	mat = scipy.io.loadmat(mat_file)
	points = mat2points(mat)
	np.save(points_file, points)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'usage: python mat2vox.py path/to/input.mat path/to/output.npy'
		exit(0)

	mat_path = sys.argv[1]
	points_path = sys.argv[2]
	matfile2pointsfile(mat_path, points_path)
	
	
