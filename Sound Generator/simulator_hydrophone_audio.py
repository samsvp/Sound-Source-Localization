# %%
import numpy as np


class AudioGenerator:

    def __init__(self, coord, fs, num_samples):
        self.fs = fs
        self.num_samples = num_samples # Number of samples to read
        
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

        # Shape (num_samples // 2, hydrophone_number, theta.shape, phi.shape)	
        self.freq_delays = np.array([np.exp(2j * np.pi * fs * deltas * k / num_samples) 
                            for k in range(num_samples // 2)]) 	
    
    def create_sine_wave(self, f=5000) -> np.ndarray:
        """
        Creates a sine wave with 'num_samples' points with the given frequency
        """
        t = self.num_samples / self.fs # duration of signal (in seconds)
        samples = np.linspace(0, t, self.num_samples, endpoint=False)
        signal = np.sin(2 * np.pi * f * samples)
        return signal

    def shift_signal(self, sine_wave: np.ndarray, azimuth: int, elevation: int) -> np.ndarray:
        """
        Shifts the signal given the azimuth and elevation angles
        """
        signal = np.fft.fft(sine_wave, axis = 0)[:self.num_samples // 2, :]
        shifted_signal_f = signal * self.freq_delays[:, :, azimuth, elevation] 
        shifted_signal = np.fft.ifft(shifted_signal_f, axis=0).real
        return shifted_signal

    def create_signals(self, azimuth: int, elevation: int, f=5000) -> np.ndarray:
        sine_wave = self.create_sine_wave(f)
        sine_waves = np.array([sine_wave for i in range(4)]).T
        shifted_signal = self.shift_signal(sine_waves, azimuth, elevation)
        return shifted_signal

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    distance_x = (19.051e-3)/2  # Distance between hydrophones in m
    distance_y = (18.37e-3)/2

    coord = np.array(([-distance_x, -8.41e-3, -distance_y],
                    [distance_x, 0, -distance_y],
                    [distance_x, -8.64e-3, distance_y],
                    [-distance_x, -0.07e-3, distance_y]
                ))

    a = AudioGenerator(coord, 192000, 128)
    shifted_signal = a.create_signals(120, 60)
    plt.plot(shifted_signal)
# %%
