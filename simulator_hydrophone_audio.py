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
    
    def create_sine_wave(self, f=5000) -> np.ndarray:
        """
        Creates a sine wave with 'num_samples' points with the given frequency
        """
        t = self.num_samples / self.fs # duration of signal (in seconds)
        samples = np.linspace(0, t, self.num_samples//2, endpoint=False)
        signal = np.sin(2 * np.pi * f * samples)
        return signal

    def add_noise(self, signal: np.ndarray, target_snr_db=10) -> np.ndarray:
        """
        Adds noise to a given signal given a target Signal to noise ration.
        Returns the signal with noise.
        For more info:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio
        https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python/53688043#53688043
        """
        # Calculate signal power and convert to dB 
        sig_avg_watts = np.mean(signal**2)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
        # Noise up the original signal
        noisy_signal = signal + noise
        return noisy_signal

    def interpolate(self, y:np.ndarray) -> CubicSpline:
        """
        Returns the cubic spline for a given signal y
        """
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

    def create_signals(self, azimuth: int, elevation: int, f=5000, add_noise=False, target_snr_db=10) -> np.ndarray:
        """
        Simulates signals that would be recorded by the hydrophones with the 
        given azimuth and elevation angles of the sound source relative to the
        hydrophone array.
        """
        sine_wave = self.create_sine_wave(f)
        shifted_signal = self.shift_signal(sine_wave, azimuth, elevation).T
        if add_noise:
            shifted_noisy_signal = np.array([self.add_noise(signal, target_snr_db) for signal in shifted_signal])
            return shifted_noisy_signal
        else:
            return shifted_signal

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
    r = range(20, 161)
    for i in r:
        az = i
        el = i
        y = a.create_signals(az, el, f=5000, add_noise=True, target_snr_db=20)
        angle = b.fast_faoa(y)
        errs.append((az-angle[0], el-angle[1]))
    plt.plot(r, errs)
# %%
