import os
import csv
import time
import numpy as np
import soundfile as sf
import multiprocessing as mp

from typing import Any, Dict, List, Tuple


class Bf:

	def __init__(self, coord, fs, num_samples, phi=(0,181), theta=(0,361), time_skip=8):
		sound_speed  = 343 # m/s
		
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

		# Apply the frequency domain beamforming
		az_offset, el_offset = (a[0] for a in self.aoa(signal, bf=self.fdsb, delays=delays, batch_size=1))

		azimuth, elevation = (az_lower_bound + az_offset, el_lower_bound + el_offset)

		return azimuth, elevation


def time_to_index_2019(filepath: str, fs: int) -> List[Tuple[int, int]]:
    dt = 1 / fs
    with open(filepath) as f:
        reader = csv.reader(f)
        indexes = [(int(float(row[1]) // dt), int(float(row[2]) // dt))
            for i, row in enumerate(reader) if i != 0]
    return indexes

def get_csv_file(wavfile: str, dev=True) -> str:
    name_split = wavfile.split("/")
    fl = name_split[-1].split(".")[0]
    csvfile = f"2019/metadata_dev/{fl}.csv" if dev else f"/content/drive/MyDrive/beamforming/metadata_eval/{fl}.csv"
    return csvfile

def get_gab(csv_file:str, gab: Dict[Any, Any]) -> Dict[str, Tuple[int, int, str, int]]:
    csv_name = csv_file.split("/")[-1].split(".")[0]
    gab[csv_name] = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        gab[csv_name].update({i-1: (180 - int(row[-2]), 90 - int(row[-3]), row[0], int(row[-1]))
        for i, row in enumerate(reader) if i != 0})


def avg_beamform(sound: np.ndarray) -> Tuple[int, int]:
    def moving_average(x: np.ndarray, N: int) -> np.ndarray:
        return np.convolve(x, np.ones(N)/N, mode='same')

    p = 0
    window = num_samples
    res = []
    
    while True:
        sound_window = sound[p:p + window ,:]
        res.append(b.fast_faoa(sound_window))
        p += window
        if p + window > len(sound):
            break

    values0, count0 = np.unique([r[0] for r in res], return_counts=1)
    values1, count1 = np.unique([r[1] for r in res], return_counts=1)
    count0 = moving_average(count0, 10)
    count1 = moving_average(count1, 10)
    return (values0[np.argmax(count0)], values1[np.argmax(count1)])


def get_error(dt_2019):
    start = time.time()

    m_class = "speech"

    y, fs = sf.read(dt_2019)
    csv_dt_2019 = get_csv_file(dt_2019)
    csv_name = csv_dt_2019.split("/")[-1].split(".")[0]

    indexes = time_to_index_2019(csv_dt_2019, fs)
    sounds = [y[idx[0]:idx[1], :] for i, idx in enumerate(indexes)]
    angle_errors = []

    get_gab(csv_dt_2019, gab)
    for i, sound in enumerate(sounds):
        class_name = gab[csv_name][i][2]
        if class_name != m_class: continue

        angles = avg_beamform(sound)
        error = [angles[0]-gab[csv_name][i][0], angles[1]-gab[csv_name][i][1]]

        if np.abs(error[0]) > 180:
            error[0] = error[0] - np.sign(error[0])*360
        if np.abs(error[1]) > 90:
            error[1] = error[1] - np.sign(error[1])*180
        angle_errors.append((angles, error))

    print(f"Processed {i} sounds from {csv_name} in {time.time() - start}")
    return angle_errors


def get_errors() -> List[Any]:
    dataset_2019 = [f"2019/mic_dev/{f}" for f in os.listdir("2019/mic_dev")]
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(get_error, dataset_2019[:100])
    return results

coord = np.array(([-0.02432757, -0.02432757, -0.02409021],
    [-0.02432757, 0.02432757, 0.02409021],
    [ 0.02432757, -0.02432757, 0.02409021],
    [ 0.02432757, 0.02432757, -0.02409021]))

fs = 48000
num_samples = 256
b = Bf(coord, fs, num_samples)

gab = {}

if __name__ == "__main__":
    results = get_errors()
    
    import pickle
    with open("out_filtered.txt", "wb") as f:
        pickle.dump(results, f)