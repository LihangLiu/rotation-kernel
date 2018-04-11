from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import random

def pltVoxes(vox, percentile=0.5):
	MAX = np.max(np.abs(vox))
	vox = vox / MAX
	vox[np.abs(vox) < percentile] = 0

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = vox.shape[0] - 1
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	xs, ys, zs = np.nonzero((vox>0) * vox)
	rgbs = np.ones((xs.shape[0],3))
	rgbs[:, 0] = 1 - vox[xs, ys, zs]
	ax.scatter(xs, ys, zs, color=rgbs)

	xs, ys, zs = np.nonzero((vox<0) * vox)
	rgbs = np.ones((xs.shape[0],3))
	rgbs[:, 1] = 1 + vox[xs, ys, zs]
	ax.scatter(xs, ys, zs, color=rgbs)


def pltPoints(pointname,N):
	print(points.shape)

	# draw 
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect("equal")
	dim = N
	ax.set_xlim(0, dim)
	ax.set_ylim(0, dim)
	ax.set_zlim(0, dim)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	xs,ys,zs = points[:,0],points[:,1],points[:,2]
	rgbs = np.full((points.shape[0],3), 0.5)
	ax.scatter(xs, ys, zs, color=rgbs) #, s=5)

	plt.show()
	

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('usage: python mat2vox.py N path/to/points.npy')
		exit(0)

	N = int(sys.argv[1])
	points_file = os.path.abspath(sys.argv[2])
	points = np.load(points_file)
	pltPoints(points,N)






