
from renderutils import progress

import voxelizer as vox
import matplotlib.image as mpimg
import numpy as np
import sys
import glob
import os
# import osp

def rot(t, p=0.0):
	theta = t
	phi = p

	sin_theta = np.sin(theta)
	cos_theta = np.cos(theta)
	sin_phi = np.sin(phi)
	cos_phi = np.cos(phi)

	ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
	rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]

	return ry


def grid_coord(h, w, d):
	xl = np.linspace(-1.0, 1.0, w)
	yl = np.linspace(-1.0, 1.0, h)
	zl = np.linspace(-1.0, 1.0, d)

	xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
	g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
	return g


def transform_volume(v, t):
	height = int(v.shape[0])
	width = int(v.shape[1])
	depth = int(v.shape[2])
	grid = grid_coord(height, width, depth)

	xs = grid[0, :]
	ys = grid[1, :]
	zs = grid[2, :]

	idxs_f = np.transpose(np.vstack((xs, ys, zs)))
	idxs_f = idxs_f.dot(t)

	xs_t = (idxs_f[:, 0] + 1.0) * float(width) / 2.0
	ys_t = (idxs_f[:, 1] + 1.0) * float(height) / 2.0
	zs_t = (idxs_f[:, 2] + 1.0) * float(depth) / 2.0

	idxs = np.vstack((xs_t, ys_t, zs_t)).astype('int')
	idxs = np.clip(idxs, 0, 31)

	return np.reshape(v[idxs[0,:], idxs[1,:], idxs[2,:]], v.shape)


def project(v, tau=1):
	p = np.sum(v, axis=2)
	p = np.ones_like(p) - np.exp(-p*tau)
	return np.flipud(np.transpose(p))


if __name__ == '__main__':
	category_path = os.getcwd() + "/ModelNet10"
	categories_npy = []
	for x in os.listdir(category_path):
	    if x[-4:]=="_npy":
	        categories_npy.append(x)
	# categories_npy = [x if x[-4:]=="_npy" for x in os.listdir(category_path)]
	print(categories_npy)
	
	for category_npy in categories_npy:
		category_png_path = os.getcwd() + "/data/" + category_npy[:-4]
		try:
			os.mkdir(category_png_path)
		except OSError as error:
			print(error)
		
		npy_paths = glob.glob("ModelNet10/" + category_npy + "/*.npy")
		npy_files = os.listdir(category_path + "/" + category_npy)
		# voxels_path = glob.glob(os.getcwd() + sys.argv[1])
		# print(sys.argv[1])
		print ("Creates dataset from volume files. Example: python create_dataset.py \"chair_volumes/*\"")
		print ("Creating dataset from ", "ModelNet10/" + category_npy)
	
		count = 0
		total = len(npy_paths)
		print("total:", total)
		for j, p in enumerate(npy_paths):
			vs = np.load(p)
			# path = p.split('.')[0]
			i = 0
			for t in np.linspace(0, 2*np.pi, 9):
				img = project(transform_volume(vs, rot(t)), 10)
				# imgpath = "{}.v{}.png".format(path, i)
				name = npy_files[j].split('.')[0]
				imgpath = category_png_path + "/" + name + ".v" + str(i) + ".png"
				# print(imgpath)
				mpimg.imsave(imgpath, img, cmap='gray')
				# mogrify_path = "/content/gdrive/My\ Drive/8sem/CS\ 726/Project/PrGAN/data/"+category_npy[:-4]
				# os.system("mogrify -alpha off {}".format(mogrify_path))
				i+=1
			progress(count, total)
			count += 1
	print ("Done.")
