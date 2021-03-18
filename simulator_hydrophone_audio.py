# %%
import numpy as np
import beamforming as bf
from scipy.interpolate import CubicSpline

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

        time_delays = np.swapaxes(
				(fs * delays).T, 1, 2
			) # Shape (phi.shape, theta.shape, hydrophone_number)

        self.time_delays = time_delays + np.max(np.abs(time_delays))
        self.n_max = int(np.max(np.abs(self.time_delays)))

        # Frequency delays
        # Shape (hydrophone_number, theta.shape, phi.shape)	
        self.deltas = np.moveaxis(delays, 0, -1) 
    
    def create_sine_wave(self, f=5000) -> np.ndarray:
        """
        Creates a sine wave with 'num_samples' points with the given frequency
        """
        t = self.num_samples / self.fs # duration of signal (in seconds)
        samples = np.linspace(0, t, self.num_samples//2, endpoint=False)
        signal = np.sin(2 * np.pi * f * samples)
        return signal

    def interpolate(self, y:np.ndarray) -> CubicSpline:
        x = np.arange(len(y))
        cs = CubicSpline(x, y)
        return cs

    def shift_signal(self, sine_wave: np.ndarray, azimuth: int, elevation: int) -> np.ndarray:
        """
        Shifts the signal given the azimuth and elevation angles
        """
        x = np.arange(len(sine_wave))
        cs = self.interpolate(sine_wave)
        shifted_signal = [cs(x - delay) for delay in self.time_delays[azimuth, elevation, :]]
        return np.array(shifted_signal)

    def create_signals(self, azimuth: int, elevation: int, f=5000) -> np.ndarray:
        sine_wave = self.create_sine_wave(f)
        shifted_signal = self.shift_signal(sine_wave, azimuth, elevation)
        return shifted_signal.T

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    distance_x = (19.051e-3)/2  # Distance between hydrophones in m
    distance_y = (18.37e-3)/2

    coord = np.array(([-distance_x, -8.41e-3, -distance_y],
                    [distance_x, 0, -distance_y],
                    [distance_x, -8.64e-3, distance_y],
                    [-distance_x, -0.07e-3, distance_y]
                ))

    fs = 192000
    num_samples = 256
    b = bf.Bf(coord, fs, num_samples//2)
    a = AudioGenerator(coord, fs, num_samples)

    errs = []
    for i in range(30,150):
        az = i
        el = i
        y = a.create_signals(az, el, f=5000)
        angle = b.fast_faoa(y)
        errs.append((az-angle[0], el-angle[1]))
    plt.plot(errs)
# %%
