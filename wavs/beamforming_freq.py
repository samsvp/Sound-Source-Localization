#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import sys

import time

import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt


class beamforming:

    def __init__(self, fs, amount_to_read, theta_resolution=1, phi_resolution=1):

        sound_speed = 1491.24  # m/s
        
        distance_x = (19.051e-3)/2  # Distance between hydrophones in m
        distance_y = (18.37e-3)/2
        #distance_z = -13.26e-3
        # XY matrix of the hydrophone coordinates
        # [X,Z,Y]
        # eixo z negativo
        coord = np.array(([-distance_x, -8.41e-3, -distance_y],
                          [distance_x, 0, -distance_y],
                          [distance_x, -8.64e-3, distance_y],
                          [-distance_x, -0.07e-3, distance_y]))
        
  
        
        # Delay matrix builder
        phi = np.deg2rad(np.arange(0, 181, phi_resolution))
        theta = np.deg2rad(np.arange(0, 181, theta_resolution))
        phi.shape, theta.shape = (1, phi.shape[0]), (theta.shape[0], 1)

        spherical_coordinates = np.concatenate(
            (
                [-np.cos(theta)*np.sin(phi)],
                [np.sin(
                    theta)*np.sin(phi)],
                [np.ones(
                    theta.shape)*np.cos(phi)]
            )
        )

        deltas = np.array(
            [np.dot(coord, spherical_coordinates[..., i])
             for i in range(spherical_coordinates.shape[-1])]
        )/sound_speed

        # Shape (hydrophone_number, theta.shape, phi.shape)
        deltas = np.moveaxis(deltas, 0, -1)

        self.delays = np.array([np.exp(2j * np.pi * fs * deltas * k / amount_to_read)
                                for k in range(int(amount_to_read / 2))])

        self.phi, self.theta = phi, theta

        self.points_to_read = int(amount_to_read / 2)

    def beamforming(self, signal):

        FFT_sinal = np.fft.fft(signal, axis=0)

        # The vectorized implementation uses all my memory, it will probably be doable and faster on a GPU
        # FFT_sinal = FFT_sinal[0:self.points_to_read,:,None,None]
        # frequency_convolution = (np.fft.ifft(FFT_sinal * self.delays, axis=0)).real

        FFT_sinal = FFT_sinal[0:self.points_to_read, :, None, None]
        frequency_convolution = np.empty(
            (self.points_to_read, 4, self.phi.shape[1], self.theta.shape[0]))

        # Find the batch size that works best on the robot
        batch_size = 8
        ps = self.phi.shape[1]
        length = int(np.ceil(ps/batch_size))
        for b in range(batch_size):
            frequency_convolution[:, :, :, b*length:b*length+length] = (np.fft.ifft(
                FFT_sinal * self.delays[:, :, :, b*length:b*length+length], axis=0)).real

        channels_sum = frequency_convolution.sum(1)
        data = (channels_sum ** 2).sum(0)

        return data

#
#
#


def main(file):
    y, fs = sf.read(file)

    y_ref = y[:, 4]  # Only for testing
    y = y[:, :4]
    y_ref = y

    amount_to_read = 256

    block_beginning_point = 0
    block_ending_point = amount_to_read

    #resolution = 1

    #data = np.array([]).reshape(int(np.ceil(181/resolution)),0)

    b = beamforming(fs, amount_to_read)

    while block_ending_point < y.shape[0]:

        signal_block = y_ref[block_beginning_point:block_ending_point, :]

        # Check if a signal was acquired
        if signal_block[signal_block > 0.01].shape[0] == 0:
            block_beginning_point += amount_to_read
            block_ending_point += amount_to_read
            continue

        signal_beginning_point = block_beginning_point - 100 + \
            np.where(signal_block ==
                     signal_block[signal_block > 0.01][0])[0][0]

        block_beginning_point += int(fs/8)
        block_ending_point = block_beginning_point + amount_to_read

        if block_ending_point > y.shape[0]:
            continue

        signal = y[signal_beginning_point:signal_beginning_point +
                   amount_to_read, :]

        start = time.time()
        rms = b.beamforming(signal)
        end = time.time()
        print("time elapsed: "+str(end - start))

        plt.imshow(rms.T, extent=[0, 181, 181, 0], aspect='auto')
        #print(np.unravel_index(rms.argmax(), rms.shape))
        azimuth, elevation = np.where(rms == np.max(rms))
        print("azimuth: " + str(azimuth) + " , elevation: " + str(elevation))
        plt.xlabel("Azimuth")
        plt.ylabel("Elevation")
        plt.colorbar()
        plt.show()
        break


if __name__ == '__main__':
    if(len(sys.argv) == 2):
        main(sys.argv[1])
    else:
        print('usage: python beamform_freq.py file_name.WAV')
