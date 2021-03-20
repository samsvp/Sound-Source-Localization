#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import time

import numpy as np
import soundfile as sf

import beamforming as bf
import visual_beamforming as vbf

#%%
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

# coord = np.array(([-distance_x, -8.41e-3, -distance_y],
#                          [distance_x, 0, -distance_y],
#                          [distance_x, -8.64e-3, distance_y],
#                          [-distance_x, -0.07e-3, distance_y]))

gab_030719 = {
    "2": (90, 90, 25),"3": (120, 90, 25),"4": (150, 90, 25),"6": (60, 90),"7": (30, 90, 25),
    "8": (30, 90),"9": (60, 90),"10": (90, 90),"11": (120, 90),"12": (150, 90),
    "13": (150, 120),"14": (120, 120),"15": (90, 120),"16": (60, 120),"17": (30, 120),
    "18": (30, 120, 25), "19": (60, 120, 25), "20": (90, 120, 25), "21": (120, 120, 25), "22": (150, 120, 25),
}

gab_110118 = {
    "1": (90, 90, 25),"2": (120, 90, 25),"3": (135, 90, 25),"4": (150, 90),"5": (165, 90, 25),
    "6": (60, 90),"7": (45, 90),"8": (30, 90),"9": (15, 90),"10": (15, 90),
    "11": (180, 90), "12": (90, 100), "13": (90,120), "14": (90, 135),"15": (120, 135),"16": (135, 135), "17": (150, 135),
    "18": (60, 120, 25), "19": (45, 120, 25), "20": (30, 120, 25)
}

amount_to_read = 128
fs = 192000
b = bf.Bf(coord, fs, amount_to_read)
err = []
for i in range(2,21):
	# if gab_030719[str(i)][0] > 150 or gab_030719[str(i)][0] < 30: continue
	n =f"0{i}" if i < 10 else str(i) 
	if i == 5: continue
	y, fs = sf.read(f'/home/samuel/Sound-Source-Localization/wavs/110118_0{n}.WAV')
	y = y[:,:4]	
	y_ref = y																			


	block_beginning_point = 0
	block_ending_point = amount_to_read

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

		if len(signal) == 0: continue

		start = time.time()
		angle = b.fast_faoa(signal)
		print(i)
		print("pure frequency fast aoa time", time.time()-start)
		print("freq fast aoa angles:", angle)
		
		err.append(angle[0] - gab_030719[str(i)][0])
		# z.append(angle[0])
		# # squared_conv = b.fdsb(signal, delays=b.fast_freq_delays)
		# # vbf.plot_squared_conv(squared_conv, show=True)

		# # squared_conv = b.dsb(signal, delays=b.fast_time_delays)
		# # vbf.plot_squared_conv(squared_conv, show=True)

		break

plt.plot(err, "o-")
# %%
