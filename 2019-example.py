#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from math import cos,sin, tan,radians
import time
import pickle
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import beamforming as bf
import visual_beamforming as vbf

#%%
# azi, el, radius
cis = np.array(([45 , 35 , 4.2*10**-2 ],
                  [-45 , -35 , 4.2*10**-2 ],
                  [135 , -35 , 4.2*10**-2 ],
                  [-135 , 35 , 4.2*10**-2 ]))
				  
coord = np.array([[	cis[0,2]*cos(radians(cis[0,0]))*sin(radians(cis[0,1])),
					cis[0,2]*sin(radians(cis[0,0]))*sin(radians(cis[0,1])),
					cis[0,2]*cos(radians(cis[0,1]))],[
					cis[1,2]*cos(radians(cis[1,1]))*sin(radians(cis[1,1])),
					cis[1,2]*sin(radians(cis[1,1]))*sin(radians(cis[1,1])),
					cis[1,2]*cos(radians(cis[1,1]))],[
					cis[2,2]*cos(radians(cis[2,1]))*sin(radians(cis[2,1])),
					cis[2,2]*sin(radians(cis[2,1]))*sin(radians(cis[2,1])),
					cis[2,2]*cos(radians(cis[2,1]))],[
					cis[3,2]*cos(radians(cis[3,1]))*sin(radians(cis[3,1])),
					cis[3,2]*sin(radians(cis[3,1]))*sin(radians(cis[3,1])),
					cis[3,2]*cos(radians(cis[3,1]))	]])

amount_to_read = 128
fs = 192000
b = bf.Bf(coord, fs, amount_to_read)

# if gab_030719[str(i)][0] > 150 or gab_030719[str(i)][0] < 30: continue 
with open(f"2019_data/split3_ir1_ov1_29.pickle", "rb") as f:
	y = pickle.load(f)
	y = y[2]
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
	print("freq fast aoa angles:", angle)
	
	
	#err.append(angle[0] - gab_030719[str(i)][0])
	# z.append(angle[0])
	# # squared_conv = b.fdsb(signal, delays=b.fast_freq_delays)
	# # vbf.plot_squared_conv(squared_conv, show=True)

	# # squared_conv = b.dsb(signal, delays=b.fast_time_delays)
	# # vbf.plot_squared_conv(squared_conv, show=True)

	break

#plt.plot(err, "o-")
# %%
