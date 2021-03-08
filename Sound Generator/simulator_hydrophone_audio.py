import numpy as np
class AudioGenerator:

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

		# Frequency delays
		# Shape (hydrophone_number, theta.shape, phi.shape)	
		deltas = np.moveaxis(delays, 0, -1) 

		# Don't compute the delays for every angle when using fast beamforming
		self.time_skip = time_skip

		# Shape (num_samples // 2, hydrophone_number, theta.shape, phi.shape)	
		self.freq_delays = np.array([np.exp(2j * np.pi * fs * deltas * k / num_samples) 
							for k in range(num_samples // 2)]) 	
		self.num_samples = num_samples  # Number of samples to read
    
    def create_sine_wave(self):
        f = 2000 # frequency 
        t = 128//self.fs # duration of signal (in seconds)
        samples = np.linspace(0, t, 128, endpoint=False)
        signal = np.sin(2 * np.pi * f * samples)
        return signal

    def shift_signal(self,theta,phi):

        sine_wave = create_sine_wave()
        signal = np.fft.fft(sine_wave, axis = 0)
        conv_signal_f = signal * freq_delays[:,:,theta,phi] 
        conv_signal = np.fft.ifft(conv_signal_f,axis=0).real()
