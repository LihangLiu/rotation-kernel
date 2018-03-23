
import numpy as np
import scipy.io
import os
import sys
import glob

from mat2points import matfile2pointsfile


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'usage: python mat2vox.py root/path/of/all/matfiles new/root/path'
		exit(0)

	root_path = sys.argv[1]
	new_root_path = sys.argv[2]
	new_extension = '.64.points.npy'

	for path, dirs, files in os.walk(root_path):
		for file in files:
			if file.endswith(".mat"):
				mat_file = os.path.join(path, file)
				relpath = os.path.relpath(path, root_path)
				new_path = os.path.join(new_root_path, relpath)
				npy_file = os.path.join(new_path, file + new_extension)
				if not os.path.exists(new_path):
					os.makedirs(new_path)
				if os.path.exists(npy_file):
					continue
				matfile2pointsfile(mat_file, npy_file)
				print(npy_file)

