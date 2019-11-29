#!/usr/bin/env python
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

y, fs = sf.read('wavs/110118_002.WAV')

y = y[:,:4]	

y_ref = y																																																								

amount_to_read = 256

block_beginning_point = 0
block_ending_point = amount_to_read

b = fb.fast_beamforming(coord, fs, amount_to_read)

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
	end = time.time()
	print(end - start)
	
	print(rms)

	break

