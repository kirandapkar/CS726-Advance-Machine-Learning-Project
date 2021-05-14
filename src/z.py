import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft

def get_z():
	fourier = np.zeros((50,50))
	gaussian = np.zeros((50,50))
	uniform1 = np.zeros((50,50))
	uniform2 = np.zeros((50,50))
	uniform3 = np.zeros((50,50))
	n = 250

	#gaussian in the middle
	#Parameters to set
	mu_x = 25
	variance_x = 50

	mu_y = 25
	variance_y = 50
	pi = 3.14
	#Create grid and multivariate normal
	x = np.linspace(0,50,50)
	y = np.linspace(0,50,50)
	X, Y = np.meshgrid(x,y)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X; pos[:, :, 1] = Y
	rv = multivariate_normal([mu_x, mu_y], [[variance_x, 10], [10, variance_y]])
	gaussian = rv.pdf(pos)*2*pi*math.sqrt(variance_y*variance_x)


	#uniform1
	xy_min1 = [0, 23]
	xy_max1 = [50, 26]
	u1 = np.random.uniform(low=xy_min1, high=xy_max1, size=(n,2))

	for ij in u1:
		uniform1[int(ij[0]),int(ij[1])] = 1
	#uniform2
	xy_min2 = [23, 0]
	xy_max2 = [26, 50]
	u2 = np.random.uniform(low=xy_min2, high=xy_max2, size=(n,2))

	for ij in u2:
		uniform2[int(ij[0]),int(ij[1])] = 1
	#uniform3
	xy_min3 = [0, 0]
	xy_max3 = [50, 50]
	u3 = np.random.uniform(low=xy_min3, high=xy_max3, size=(n,2))

	for ij in u3:
		uniform3[int(ij[0]),int(ij[1])] = 1

	fourier = gaussian*1.5+uniform1*1.1+uniform2*1.1+uniform3 #just reduced the height of gaussian
	#Make a 3D plot
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')

	Z = ifft(fourier)
	Z = Z.real
	z = np.zeros((50*50+1))
	z[0:50*50] = Z.flatten()
	z[50*50] = np.random.uniform(-1,1)

	# Feedforward neural network layer
	# Layer 1: Fully connected layer with tanh activation
	weights1 = np.random.rand(2501, 5000) - 0.5
	bias1 = np.random.rand(1, 5000) - 0.5
	z = np.tanh(np.dot(z,weights1) + bias1)
	
	# Layer 2: Fully connected layer with tanh activation
	weights2 = np.random.rand(5000, 201) - 0.5
	bias2 = np.random.rand(1, 201) - 0.5
	z = np.tanh(np.dot(z,weights2) + bias2)

	# print(z.shape)
	# ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
	# ax.set_xlabel('X axis')
	# ax.set_ylabel('Y axis')
	# ax.set_zlabel('Z axis')
	# plt.show()
	return z
get_z()