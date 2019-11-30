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

		# Don't compute the delays for every angle
		self.skip_amount = 5
		self.fast_time_delays = self.time_delays[::self.skip_amount, ::self.skip_amount,:]

		# Frequency delays
		deltas = np.moveaxis(delays, 0, -1) # Shape (hydrophone_number, theta.shape, phi.shape)	

		# Shape (num_samples // 2, hydrophone_number, theta.shape, phi.shape)	
		self.freq_delays = np.array([np.exp(2j * np.pi * fs * deltas * k / num_samples) 
							for k in range(num_samples // 2)]) 


	def ffb(self, signal):
		angles = self.tb(signal, self.fast_time_delays)

		el_lower_bound = self.skip_amount * min(angles[0])-self.skip_amount 
		el_upper_bound = self.skip_amount * max(angles[0])+self.skip_amount
		az_lower_bound = self.skip_amount * min(angles[1])-self.skip_amount
		az_upper_bound = self.skip_amount * max(angles[1])+self.skip_amount

		delays = self.freq_delays[:,:,
					az_lower_bound:az_upper_bound,
					el_lower_bound:el_upper_bound 
				]

		# Apply the frequency domain beamforming
		el_offset, az_offset = self.fb(signal, delays, 1)

		elevation, azimuth = (el_lower_bound + el_offset, az_lower_bound + az_offset)

		return elevation, azimuth


	def tb(self, signal, delays=None):
		"""
        Time domain beamforming.
        Convolves the given signal with the respective coordinates delay
        and returns the squared sum of the result
		"""
		if delays is None: delays = self.time_delays

		padded_signal = np.zeros((self.n_max + signal.shape[0], signal.shape[1]))
		padded_signal[self.n_max:,:] = signal

		# shifts the signal in time by each delay
		shifted_signal = [ [ [
				padded_signal[d:self.num_samples+d,idx]
				for idx,d in enumerate(dly) ]
				for dly in delay ]
				for delay in delays ]

		squared_conv = ((np.sum(shifted_signal,axis=2))**2).sum(-1)

		# angles with maximum squared sum
		angles = np.where(squared_conv == squared_conv.max())

		return angles

	
	def fb(self, signal, delays=None, batch_size=1):
		"""
        Frequency domain beamforming.
        Multiplies the signal and delays on the frequency domain
        and returns the squared sum of the ifft of the result.
		Use this version for signals with a lot of data.
		"""
		if delays is None: delays = self.freq_delays

		Signal = np.fft.fft(signal, axis=0)
		
		Signal = Signal[0:self.num_samples // 2,:,None,None]
		conv_signal = np.empty((self.num_samples // 2, delays.shape[1], 
							delays.shape[2] , delays.shape[3]))
		
		# For signals with many samples, a bigger batch size
		# can speed up the algorithm and prevent memory errors
		ps = self.phi.shape[1]
		length = int(np.ceil(ps/batch_size))

		for b in range(batch_size):
			conv_signal[...,b*length:b*length+length] = (np.fft.ifft(
				Signal * delays[...,b*length:b*length+length], 
				axis=0)).real

		squared_conv = ((conv_signal.sum(1)) ** 2).sum(0)

		azimuth, elevation = np.unravel_index(squared_conv.argmax(), squared_conv.shape)

		return elevation, azimuth
