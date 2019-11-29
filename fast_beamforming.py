#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import time
import numpy as np

class fast_beamforming:

	def __init__(self, coord, fs, num_samples):
		sound_speed  = 1491.24 # m/s
		
		# Delay matrix builder
		self.phi = np.deg2rad(np.arange(0,181))
		self.theta = np.deg2rad(np.arange(0, 181))
		self.phi.shape, self.theta.shape = (1, self.phi.shape[0]), (self.theta.shape[0], 1)

		spherical_coordinates = np.concatenate( 
									(
										[-np.cos(self.theta)*np.sin(self.phi)],
										[np.sin(self.theta)*np.sin(self.phi)],
										[np.ones(self.theta.shape)*np.cos(self.phi)]
									)
								)

		delays = np.array(
						[np.dot(coord, spherical_coordinates[...,i]) for i in range(spherical_coordinates.shape[-1])]
					)/sound_speed  # Shape (phi.shape, hydrophone_number, theta.shape)	

		# Time delay
		self.time_delays = np.swapaxes((np.round(fs * delays).T).astype(int).T, 1, 2) # Shape (phi.shape, theta.shape, hydrophone_number)
		self.num_samples = num_samples  # Number of samples to read. If not divisible by 
						# n_amostraPcanal a amount of time will not be read

		self.n_max = int(np.max(np.abs(self.time_delays)))  # Gets the maximum absolute value os the delays matrix
		self.time_delays += self.n_max # shifts the delays matrix so it doesn't have negative values

		self.n_max = int(np.max(self.time_delays)) # Update the maximum value

		# Frequency delays
		deltas = np.moveaxis(delays, 0, -1) # Shape (hydrophone_number, theta.shape, phi.shape)	

		# Shape (num_samples // 2, hydrophone_number, theta.shape, phi.shape)	
		self.freq_delays = np.array([np.exp(2j * np.pi * fs * deltas * k / num_samples) 
							for k in range(num_samples // 2)]) 


	def ffb(self, signal):
		"""
		Fast frequency domain beamforming.
		Applies the time domain beamforming and the frequency domain
		beamforming to the result to get the pinger position 
		"""
		max_squared_index = self.tb(signal)

		
		# Get the correspondent delays
		# The azimuth and elevation are switched between the time and frequency domain
		delays = self.freq_delays[:,:,
					min(max_squared_index[1]):max(max_squared_index[1]), 
					min(max_squared_index[0]):max(max_squared_index[0])
				]

		# Apply the frequency domain beamforming
		offset = self.fb(signal, delays, 1)

		return max_squared_index[0][0] + offset[0], max_squared_index[1][0] + offset[1]


	def tb(self, signal):
		"""
        Time domain beamforming.
        Convolves the given signal with the respective coordinates delay
        and returns the squared sum of the result
		"""
		padded_signal = np.zeros((self.n_max + signal.shape[0], signal.shape[1]))
		padded_signal[self.n_max:,:] = signal

		conv_signal = [ [ [
				padded_signal[value:self.num_samples+value,i]
				for i,value in enumerate(d) ]
				for d in delay ]
				for delay in self.time_delays ]

		squared_sum = ((np.sum(conv_signal,axis=2))**2).sum(-1)

		# angles with maximum squared sum
		max_angles = np.where(squared_sum == squared_sum.max())

		return max_angles

	
	def fb(self, signal, delays=None, batch_size=8):
		"""
        Frequency domain beamforming.
        Multiplies the signal and delays on the frequency domain
        and returns the squared sum of the ifft of the result.
		Use this version for signals with a lot of data.
		"""
		if delays is None: delays = self.freq_delays

		Signal = np.fft.fft(signal, axis=0)
		
		Signal = Signal[0:self.num_samples // 2,:,None,None]
		result = np.empty((self.num_samples // 2, delays.shape[1], 
							delays.shape[2] , delays.shape[3]))
		
		# Find the batch size that works best on the robot
		ps = self.phi.shape[1]
		length = int(np.ceil(ps/batch_size))

		for b in range(batch_size):
			result[...,b*length:b*length+length] = (np.fft.ifft(
				Signal * delays[...,b*length:b*length+length], 
				axis=0)).real

		squared_sum = ((result.sum(1)) ** 2).sum(0)

		az, el = np.unravel_index(squared_sum.argmax(), squared_sum.shape)

		return el, az
