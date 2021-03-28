#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import time
import numpy as np
from scipy.interpolate import CubicSpline


class Bf:

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

		time_delays_s = np.swapaxes(
				(fs * delays).T, 1, 2
			) # Shape (phi.shape, theta.shape, hydrophone_number)

		self.time_delays_s = time_delays_s + np.max(np.abs(time_delays_s))

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
				
		self.fast_time_delays_s = self.time_delays_s[::self.time_skip, ::self.time_skip,:]

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

		Signal = np.fft.fft(signal, axis=0)[0:self.num_samples // 2,:,None,None]
		
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


	def aoa(self, signal, bf, **kwargs):
		"""
		Returns the possible angles of arrival given by the beamforming algorithm
		"""
		squared_conv = bf(signal, **kwargs)
		angles = np.where(squared_conv == squared_conv.max())
		return angles


	def s_aoa(self, signal, f_weight=1, s_weight=1):
		def interpolate(y:np.ndarray) -> CubicSpline:
			"""
			Returns the cubic spline for a given signal y
			"""
			x = np.arange(len(y))
			cs = CubicSpline(x, y)
			return cs

		angles = self.aoa(signal, bf=self.fdsb, delays=self.fast_freq_delays)

		az_lower_bound = self.time_skip * min(angles[0])-self.time_skip 
		az_upper_bound = self.time_skip * max(angles[0])+self.time_skip
		el_lower_bound = self.time_skip * min(angles[1])-self.time_skip
		el_upper_bound = self.time_skip * max(angles[1])+self.time_skip

		if el_lower_bound < 0: el_lower_bound = 0
		if az_lower_bound < 0: az_lower_bound = 0

		delays = self.time_delays_s[
					az_lower_bound:az_upper_bound,
					el_lower_bound:el_upper_bound,
					:
				]

		css = [interpolate(signal[:,i]) for i in range(4)]
		x = np.arange(len(signal))
		shifted_signal = [ [ [
					css[i](x + d)
					for i,d in enumerate(dly) ]
					for dly in delay ]
					for delay in delays ]

		# print(np.array(shifted_signal).shape)
		squared_conv_stime = ((np.sum(shifted_signal,axis=2))**2).sum(-1)

		#### freq ####
		freq_delays = self.freq_delays[:,:,
					az_lower_bound:az_upper_bound,
					el_lower_bound:el_upper_bound 
				]

		squared_conv_freq = self.fdsb(signal, delays=freq_delays)

		squared_conv = s_weight * squared_conv_stime/np.max(squared_conv_stime) + f_weight * squared_conv_freq/np.max(squared_conv_freq)

		# import visual_beamforming as vbf
		# squared_conv = self.fdsb(signal, delays=delays)
		# vbf.plot_squared_conv(squared_conv, show=True)

		angles = np.where(squared_conv == squared_conv.max())
		az_offset, el_offset = angles[0], angles[1]
		return (az_lower_bound + az_offset, el_lower_bound + el_offset)
	

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
		angles = self.aoa(signal, bf=self.fdsb, delays=self.fast_freq_delays)

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


		# import visual_beamforming as vbf
		# squared_conv = self.fdsb(signal, delays=delays)
		# vbf.plot_squared_conv(squared_conv, show=True)

		# Apply the frequency domain beamforming
		az_offset, el_offset = (a[0] for a in self.aoa(signal, bf=self.fdsb, delays=delays, batch_size=1))
		# print("angles", az_offset, el_offset)
		azimuth, elevation = (az_lower_bound + az_offset, el_lower_bound + el_offset)

		return azimuth, elevation
