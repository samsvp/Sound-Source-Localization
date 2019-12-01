#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import time

import numpy as np
import soundfile as sf

import fast_beamforming as fb

# Hydrophones coordinates
distance_x = 18.75e-3
distance_y = 18.75e-3
distance_z = -10e-3
distance     = 3 * 10**-2 # Distance between hydrophones in m

# XY matrix of the hydrophone coordinates
coord = np.array((
                            [0,0,distance],
                            [0,0,0],
                            [0,0,-distance],
                            [-distance,0,0]
                ))

distance_x = (19.051e-3)/2  # Distance between hydrophones in m
distance_y = (18.37e-3)/2

coord = np.array(([-distance_x, -8.41e-3, -distance_y],
                          [distance_x, 0, -distance_y],
                          [distance_x, -8.64e-3, distance_y],
                          [-distance_x, -0.07e-3, distance_y]))

y, fs = sf.read('wavs/030719_002.WAV')

y = y[:,:4]	

y_ref = y																			
amount_to_read = 256

block_beginning_point = 0
block_ending_point = amount_to_read

b = fb.ffb(coord, fs, amount_to_read)

thresh = 0.1

while block_ending_point < y.shape[0]:
	
	signal_block = y_ref[block_beginning_point:block_ending_point, :]	
	
	if signal_block[signal_block>thresh].shape[0] == 0:  # Check if a signal was acquired
		block_beginning_point += amount_to_read
		block_ending_point += amount_to_read
		continue

	signal_beginning_point = block_beginning_point - int(amount_to_read / 2) + \
								np.where(signal_block==signal_block[signal_block>thresh][0])[0][0]

	block_beginning_point += int(fs/8)
	block_ending_point = block_beginning_point + amount_to_read

	if block_ending_point > y.shape[0]: continue

	signal = y[signal_beginning_point:signal_beginning_point + amount_to_read,:]
	
	start = time.time()
	rms = b.ffb(signal)
	print(time.time()-start)
	
	print(rms)

	# start = time.time()
	# rms = b.fdsb(signal)
	# print(time.time()-start)
	
	# print(rms)

	# start = time.time()
	# rms = b.dsb(signal)
	# print(time.time()-start)
	
	# print(np.unique(rms[0]))
	# print(np.unique(rms[1]))

	break


# times = []
# b = fb.ffb(coord, fs, amount_to_read)
# for i in range(100):
# 	start = time.time()
# 	rms = b.ffb(signal)
# 	times.append(time.time()-start)
# print("Fast beamforming (100 iterations):")
# print("mean:", np.mean(times), "\nmax:", np.max(times), "\nmin:", np.min(times))

# print("")

# times = []
# for i in range(100):
# 	start = time.time()
# 	rms = b.fdsb(signal)
# 	times.append(time.time()-start)
# print("Frequency beamforming (100 iterations):")
# print("mean:", np.mean(times), "\nmax:", np.max(times), "\nmin:", np.min(times))

# print("")

# times = []
# for i in range(100):
# 	start = time.time()
# 	rms = b.dsb(signal)
# 	times.append(time.time()-start)
# print("Time beamforming (100 iterations):")
# print("mean:", np.mean(times), "\nmax:", np.max(times), "\nmin:", np.min(times))
