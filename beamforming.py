#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import time
import numpy as np
from numpy.core.numeric import moveaxis


class Bf:

	def __init__(self, coord, fs, num_samples, speed,  phi=(0,181), theta=(0,361), time_skip=8):
		self.speed  = speed # m/s
		
		# Delay matrix builder
		self.phi = np.deg2rad(np.arange(*phi)).reshape(1, phi[-1])
		self.theta = np.deg2rad(np.arange(*theta)).reshape(theta[-1],1)

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
					)/self.speed  # Shape (phi.shape, hydrophone_number, theta.shape)	

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

		# Don't compute the delays for every angle when using fast beamforming
		self.time_skip = time_skip

		# Shape (num_samples // 2, hydrophone_number, theta.shape, phi.shape)	
		self.freq_delays = np.array([np.exp(2j * np.pi * fs * deltas * k / num_samples) 
							for k in range(num_samples // 2)]) 
		self.fast_freq_delays = self.freq_delays[:,:,::self.time_skip, ::self.time_skip]
		
		self.fast_time_delays = self.time_delays[::self.time_skip, ::self.time_skip,:]
				
		self.num_samples = num_samples  # Number of samples to read


	def dsb(self, signal, **kwargs):
		"""
        Time domain delay-and-sum beamforming.
        Convolves the given signal with the respective coordinates delay
        and returns the squared sum of the result
		"""
		delays = kwargs.get("delays", self.time_delays)

		padded_signal = np.pad(signal, ((self.n_max,0),(0,0)), mode="constant")

		# shifts the signal in time by each delay
		shifted_signal = [ [ [
				padded_signal[d:self.num_samples+d,idx]
				for idx,d in enumerate(dly) ]
				for dly in delay ]
				for delay in delays ]
		
		squared_conv = ((np.sum(shifted_signal,axis=2))**2).sum(-1).T

		return squared_conv

	
	def fdsb(self, signal, **kwargs):
		"""
        Frequency domain delay-and-sum beamforming.
        Multiplies the signal and delays on the frequency domain
        and returns the squared sum of the ifft of the result
		"""
		delays = kwargs.get("delays", self.freq_delays)
		batch_size = kwargs.get("batch_size", 1)

		Signal = np.fft.fft(signal, axis=0)[:self.num_samples // 2,:,None,None]

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

		return squared_conv


	def fast_fdsb(self, signals, delays):
		"""
		A faster implementation of fdsb
		"""
		if (len(signals.shape) == 2): signals = signals[None,:,:]

		fconv = np.einsum("kij,ijlm->ilmk", signals, delays)
		conv = np.fft.ifft(fconv, axis=0).real
		squared_conv = np.einsum("ijkm,ijkm->jkm", conv, conv)
		return squared_conv
	

	def aoa(self, signal, bf, **kwargs):
		"""
		Returns the possible angles of arrival given by the beamforming algorithm
		"""
		squared_conv = bf(signal, **kwargs)

		if len(squared_conv.shape) <= 2:
			_angles = np.where(squared_conv == squared_conv.max())
			angles = np.array(_angles)
		else:
			_angles = [
				np.where(squared_conv[...,i] == squared_conv[...,i].max()) 
				for i in range(squared_conv.shape[-1])
			]
			angles = np.array([[a[0][0], a[1][0]] for a in _angles]).T

		return angles
		
	
	def fast_aoa(self, signal):
		"""
        Fast delay-and-sum beamforming.
        Applies the time domain beamforming to the signal to get the area
		with the highest rms, then applies the frequency domain beamforming
		to get the right angles
		"""
		angles = self.aoa(signal, bf=self.dsb, delays=self.fast_time_delays)

		az_lower_bound = self.time_skip * min(angles[0])-self.time_skip 
		az_upper_bound = self.time_skip * max(angles[0])+self.time_skip
		el_lower_bound = self.time_skip * min(angles[1])-self.time_skip
		el_upper_bound = self.time_skip * max(angles[1])+self.time_skip

		if el_lower_bound < 0: el_lower_bound = 0
		if az_lower_bound < 0: az_lower_bound = 0

		delays = self.freq_delays[:,:,
					az_lower_bound:az_upper_bound,
					el_lower_bound:el_upper_bound 
				]

		# Apply the frequency domain beamforming
		az_offset, el_offset = (a[0] for a in self.aoa(signal, bf=self.fdsb, delays=delays, batch_size=1))

		azimuth, elevation = (az_lower_bound + az_offset, el_lower_bound + el_offset)

		return azimuth, elevation

	
	def fast_faoa(self, signal):
		"""
        Fast delay-and-sum beamforming.
        Applies the frequency domain beamforming to the signal to get the area
		with the highest rms, then applies the frequency domain beamforming
		to get the right angles
		"""
		angles = self.aoa(signal, bf=self.fast_fdsb, delays=self.fast_freq_delays)

		az_lower_bound = self.time_skip * min(angles[0])-self.time_skip 
		az_upper_bound = self.time_skip * max(angles[0])+self.time_skip
		el_lower_bound = self.time_skip * min(angles[1])-self.time_skip
		el_upper_bound = self.time_skip * max(angles[1])+self.time_skip

		if el_lower_bound < 0: el_lower_bound = 0
		if az_lower_bound < 0: az_lower_bound = 0

		delays = self.freq_delays[:,:,
					az_lower_bound:az_upper_bound,
					el_lower_bound:el_upper_bound 
				]

		# Apply the frequency domain beamforming
		az_offset, el_offset = (a[0] for a in self.aoa(signal, bf=self.fast_fdsb, delays=delays))

		azimuth, elevation = (az_lower_bound + az_offset, el_lower_bound + el_offset)

		return azimuth, elevation

	
	def parallel_fast_aoa(self, fsignal: np.ndarray) -> np.ndarray:
		"""
		Runs the beamforming on multiple signals at the same time
		This is an optimized version of fast_faoa to use on big
		signals
		"""
		moving_average = lambda x, N: np.convolve(x, np.ones(N)/N, mode='same')

		def get_angles(signals: np.ndarray, delays: np.ndarray) -> np.ndarray:
			"""
			Returns the angles from the frequency delay and sum beamforming
			"""
			squared_conv = self.fast_fdsb(signals, delays)

			_angles = [ np.where(squared_conv[...,i] == squared_conv[...,i].max())
				for i in range(squared_conv.shape[-1])]
			
			angles = np.array([[a[0][0], a[1][0]] for a in _angles]).T

			return angles

		def get_signal_indexes(angle: np.ndarray) -> np.ndarray:
			"""
			Applies moving average to angles count and return
			the most returned angle
			"""
			u, c = np.unique(angle, return_counts=1)
			return u[np.argmax(moving_average(c,10))]

		new_shape = (fsignal.shape[0]//self.num_samples, self.num_samples, fsignal.shape[1])
		signals = fsignal[:new_shape[0]*new_shape[1],:].reshape(*new_shape)
		Signals = np.fft.fft(signals, axis=1)[:, :self.num_samples // 2, :]

		az, el = get_angles(Signals, self.fast_freq_delays)
		
		az_lower_bound = self.time_skip * az - self.time_skip 
		az_upper_bound = self.time_skip * az + self.time_skip
		el_lower_bound = self.time_skip * el - self.time_skip
		el_upper_bound = self.time_skip * el + self.time_skip

		el_lower_bound[el_lower_bound < 0] = 0
		az_lower_bound[az_lower_bound < 0] = 0

		# fucking edge cases
		el_max_diff = np.max(el_upper_bound - el_lower_bound)
		az_max_diff = np.max(az_upper_bound - az_lower_bound)
		el_upper_bound[el_lower_bound == 0] = el_max_diff
		az_upper_bound[az_lower_bound == 0] = az_max_diff

		el_lower_bound[el_upper_bound > self.freq_delays.shape[-1]] = \
			self.freq_delays.shape[-1] - el_max_diff
		az_lower_bound[az_upper_bound > self.freq_delays.shape[-2]] = \
			self.freq_delays.shape[-2] - az_max_diff

		delays = [self.freq_delays[:,:,
			az_lower_bound[i]: az_upper_bound[i],
			el_lower_bound[i]: el_upper_bound[i] 
		] for i in range(az_lower_bound.shape[0])]

		# Apply the frequency domain beamforming
		offsets = np.array([get_angles(Signals[i:i+1,...], delays[i])
			for i in range(len(delays))])[...,0]

		return (get_signal_indexes(az_lower_bound + offsets[:,0]),
				get_signal_indexes(el_lower_bound + offsets[:,1]))

