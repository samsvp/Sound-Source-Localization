#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division


import time
import timeit

import numpy as np
import soundfile as sf

import beamforming as bf
import visual_beamforming as vbf

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

y, fs = sf.read('wavs/030719_013.WAV')

y = y[:,:4]	

y_ref = y																			
amount_to_read = 128

block_beginning_point = 0
block_ending_point = amount_to_read

b = bf.bf(coord, fs, amount_to_read)

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
	angle = b.fast_aoa(signal)
	print("normal fast anfle of arrival(aoa)", time.time()-start)
	print("fast aoa angle:", angle)

	start = time.time()
	angle = b.fast_faoa(signal)
	print("\npure frequency fast aoa time", time.time()-start)
	print("freq fast aoa angles:", angle)
	
	# squared_conv = b.fdsb(signal)
	# vbf.plot_squared_conv(squared_conv)
	# squared_conv = b.dsb(signal)
	# vbf.plot_squared_conv(squared_conv, show=True)

	start = time.time()
	angle = b.aoa(signal, b.fdsb, batch_size=8)
	print("\n normal freq aoa time:", time.time()-start)
	print("normal freq aoa angles:", angle)

	start = time.time()
	angle = b.aoa(signal, b.dsb)
	print("\nnormal time aoa time:", time.time()-start)
	
	print(np.unique(angle[0]))
	print(np.unique(angle[1]))

	break

# vbf.plot_array(coord)

# speed test
# iter_number = 100

# t = timeit.timeit("b.fast_aoa(signal)", number=iter_number, globals=globals())/iter_number
# print("\nfast aoa:", t)

# t = timeit.timeit("b.aoa(signal, b.fdsb)", number=iter_number, globals=globals())/iter_number
# print("\nfreq aoa:", t)

# t = timeit.timeit("b.aoa(signal, b.dsb)", number=iter_number, globals=globals())/iter_number
# print("\ntime aoa:", t)