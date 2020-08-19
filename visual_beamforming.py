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


def sensor_response(coords, fs=192e+3, angles=(0,90), amount_to_read = 128, show=True):
	"""
	Visualizes the sensor array response to a cossine wave
	which hits the sensors at the same time
	"""
	cos = _create_cos(fs=fs)
	bm = bf.bf(coords, fs, amount_to_read)

	delayed_cos = []
	for i in range(coords.shape[0]):
		delay = bm.time_delays[angles[0], angles[1], i]
		delayed_cos.append(cos[delay:delay + amount_to_read])

	delayed_signals = np.array(delayed_cos).reshape((amount_to_read, coords.shape[0]))

	squared_conv = bm.dsb(delayed_signals)
	plot_squared_conv(squared_conv, show=show)


def _create_cos(t=0.1, f=20e+2, fs=192e+3, A=1):
	samples = np.arange(t * fs) / fs
	signal = A * np.cos(2 * np.pi * f * samples)
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
	