from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import beamforming as bf
import matplotlib.pyplot as plt


plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True


def plot_squared_conv(squared_conv, show=False):
	"""
	Plots a heat map of the squared convolution given by the beamforming algorithm
	"""
	plt.figure()
	plt.imshow(squared_conv, extent=[0,181,181, 0])
	plt.colorbar()
	if show: plt.show()


def show():
	plt.show()


def sensor_cos_response(coords, show=False):
	"""
	Visualizes the sensor array response to a cossine wave
	which hits the sensors at the same time
	"""
	fs = 1.9e+6
	signal = np.tile(_create_cos(fs=fs), (4,1))
	b = bf.bf(coords, fs, signal.shape[0])
	squared_conv = b.dsb(signal)
	plot_squared_conv(squared_conv, show=show)


def _create_cos(t=0.1, f=20e+3, fs=1.9e+6):
	samples = np.arange(t * fs) / fs
	signal = 5 * np.cos(2 * np.pi * f * samples)
	return signal


def plot_array(coords):
	"""
	Plots the sensor coordinates
	""" 
	fig = plt.figure()
	ax = Axes3D(fig)

	for c in coords:
			ax.scatter(c[0],c[1],c[2])
	
	plt.show()
	