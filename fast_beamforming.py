#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import time
import numpy as np

class ffb:

	def __init__(self, coord, fs, num_samples, time_skip=8):
		sound_speed  = 1491.24 # m/s
		
		# Delay matrix builder
		self.phi = np.deg2rad(np.arange(0,181)).reshape(1,181)
		self.theta = np.deg2rad(np.arange(0, 181)).reshape(181,1)

		spherical_coords = np.concatenate( 
									(
										[-np.cos(self.theta)*np.sin(self.phi)],
										[np.sin(self.theta)*np.sin(self.phi)],
										[np.ones(self.theta.shape)*np.cos(self.phi)]
									)
								)

		delays = np.array(
						[np.dot(coord, spherical_coords[...,i]) 
						for i in range(spherical_coords.shape[-1])]
					)/sound_speed  # Shape (phi.shape, hydrophone_number, theta.shape)	

		# Time delay
		time_delays = np.swapaxes(
				(np.round(fs * delays).T).astype(int).T, 1, 2
			) # Shape (phi.shape, theta.shape, hydrophone_number)
		
		# Gets the maximum absolute value os the delays matrix
		n_max = int(np.max(np.abs(time_delays)))  
		
		# shifts the delays matrix so it doesn't have negative values
		self.time_delays = time_delays + n_max 
		# Update the maximum value
		self.n_max = int(np.max(self.time_delays)) 

		# Frequency delays
		# Shape (hydrophone_number, theta.shape, phi.shape)	
		deltas = np.moveaxis(delays, 0, -1) 

		# Shape (num_samples // 2, hydrophone_number, theta.shape, phi.shape)	
		self.freq_delays = np.array([np.exp(2j * np.pi * fs * deltas * k / num_samples) 
							for k in range(num_samples // 2)]) 
		
		# Don't compute the delays for every angle when using fast beamforming
		self.time_skip = time_skip
		self.fast_time_delays = self.time_delays[::self.time_skip, ::self.time_skip,:]

		self.num_samples = num_samples  # Number of samples to read


	def ffb(self, signal):
		"""
        Fast delay-and-sum beamforming.
        Applies the time domain beamforming to the signal to get the area
		with the highest rms, then applies the frequency domain beamforming
		to get the right angles
		"""

		angles = self.dsb(signal, self.fast_time_delays)

		el_lower_bound = self.time_skip * min(angles[0])-self.time_skip 
		el_upper_bound = self.time_skip * max(angles[0])+self.time_skip
		az_lower_bound = self.time_skip * min(angles[1])-self.time_skip
		az_upper_bound = self.time_skip * max(angles[1])+self.time_skip

		if el_lower_bound < 0: el_lower_bound = 0
		if az_lower_bound < 0: az_lower_bound = 0

		delays = self.freq_delays[:,:,
					az_lower_bound:az_upper_bound,
					el_lower_bound:el_upper_bound 
				]

		# Apply the frequency domain beamforming
		el_offset, az_offset = self.fdsb(signal, delays, 1)

		elevation, azimuth = (el_lower_bound + el_offset, az_lower_bound + az_offset)

		return elevation, azimuth


	def dsb(self, signal, delays=None):
		"""
        Time domain delay-and-sum beamforming.
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

	
	def fdsb(self, signal, delays=None, batch_size=1):
		"""
        Frequency domain delay-and-sum beamforming.
        Multiplies the signal and delays on the frequency domain
        and returns the squared sum of the ifft of the result
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
