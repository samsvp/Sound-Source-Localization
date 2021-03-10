#!/usr/bin/env python3
# %%
import numpy as np
import beamforming as bf
import visual_beamforming as vbf


class AudioGenerator:

    def __init__(self, coord, fs, num_samples):
        self.fs = fs
        self.num_samples = num_samples  # Number of samples to read

        sound_speed = 1491.24  # m/s

        # Delay matrix builder
        self.phi = np.deg2rad(np.arange(0, 181)).reshape(1, 181)
        self.theta = np.deg2rad(np.arange(0, 181)).reshape(181, 1)

        spherical_coords = np.concatenate(
            (
                [-np.cos(self.theta)*np.sin(self.phi)],
                [np.sin(self.theta)*np.sin(self.phi)],
                [np.ones(self.theta.shape)
                 * np.cos(self.phi)]
            )
        )

        delays = np.array(
            [np.dot(coord, spherical_coords[..., i])
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
        t = self.num_samples / self.fs  # duration of signal (in seconds)
        samples = np.linspace(0, t, self.num_samples, endpoint=False)
        signal = np.sin(2 * np.pi * f * samples)
        return signal

    def shift_signal(self, sine_wave: np.ndarray, azimuth: int, elevation: int) -> np.ndarray:
        """
        Shifts the signal given the azimuth and elevation angles
        """
        signal = np.fft.fft(sine_wave, axis=0)[:self.num_samples // 2, :]
        shifted_signal_f = signal * self.freq_delays[:, :, azimuth, elevation]
        shifted_signal = np.fft.ifft(shifted_signal_f, axis=0).real
        return shifted_signal

    def create_signals(self, azimuth: int, elevation: int, f=5000) -> np.ndarray:
        sine_wave = self.create_sine_wave(f)
        sine_waves = np.array([sine_wave, sine_wave, sine_wave, sine_wave]).T
        shifted_signal = self.shift_signal(sine_waves, azimuth, elevation)
        shifted_signal = self.create_zeroes(shifted_signal)
        return shifted_signal, self.fs

    def create_zeroes(self, shifted_signal: np.ndarray) -> np.ndarray:

        for i in range(self.num_samples//4):
            for j in range(len(shifted_signal[i])):
                shifted_signal[i, j] = 0

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

    a = AudioGenerator(coord, 192000, 256)
    shifted_signal, fs = a.create_signals(120, 130)
    plt.plot(shifted_signal)
    plt.show()

    b = bf.Bf(coord, fs, 128)

    angle = b.fast_aoa(shifted_signal)
    print("fast aoa angle:", angle)

    angle = b.fast_faoa(shifted_signal)
    print("freq fast aoa angles:", angle)

    angle = b.aoa(shifted_signal, b.fdsb, batch_size=8)
    print("normal freq aoa angles:", angle)

    angle = b.aoa(shifted_signal, b.dsb)
    print("normal time aoa angles:", angle)

    squared_conv = b.dsb(shifted_signal)
    vbf.plot_squared_conv(squared_conv, show=True)

# %%
