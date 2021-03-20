import random
import numpy as np
from simulator_hydrophone_audio import AudioGenerator

from typing import List, Tuple

def create_audio_generator(coord: np.ndarray, fs: int, num_samples: int) -> AudioGenerator:
    audio_generator = AudioGenerator(coord, fs, num_samples)
    return audio_generator

def generate_training_set(size: int, audio_generator: AudioGenerator, angles_range=(30,150),
                          f=5000, add_noise=True, target_snr_db=10) -> \
                          Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    y = []
    X = []
    append_X = X.append # for better performance
    append_y = y.append
    for _ in range(size):
        az, el = random.randint(*angles_range), random.randint(*angles_range)
        append_y((az, el))

        signals = audio_generator.simulate_signals(az, el, f=f, add_noise=add_noise, target_snr_db=target_snr_db)
        append_X(signals)

    return X, y
