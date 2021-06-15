# WIP Error Correction Beamforming
We are currently working on ways to get better angle of arrival results using the beamforming with error correction techniques. The initial results can be found on this [colab](https://colab.research.google.com/drive/128x7nEmxwqgsZOJ4lMjV5xkMgLnW3B4C?usp=sharing)

# Fast Angle of Arrival

The fast angle of arrival is an implementation of the frequency domain beamforming and time domain beamforming algorithms to localizate one sound source in an environment.

# Table of Contents
1. [Frequency Domain Beamforming](#Frequency-Domain-Beamforming)
2. [Time Domain Beamforming](#Time-Domain-Beamforming)
3. [Fast Angle of Arrival](#Fast-Angle-of-Arrival)
4. [Speed Test](#Speed-Test)
5. [Usage](#Usage)

## Frequency Domain Beamforming

The frequency domain beamforming algorithm can be used to return the azimuth and elevation angles of a signal source given an array of sensors that capture the signal.

The frequency beamforming works by multiplying the fast fourier transform (FFT) of the signal given by each sensor with the FFT of the delays matrix associated with the sensors position array and the angles to look for the source, taking the inverse FFT of the result and summing along the hydrophones axis. The final azimuth and elevation angles is given by getting the argmax of the squared sum of result.


## Time Domain Beamforming

The time domain beamforming algorithm can be used to return a set of azimuth and elevation angles of a signal source given an array of sensors that capture the signal.

The time beamforming works by padding the given sensor signals with zeros and shifting the signal in time by the delays matrix associated with the sensors position array and the angles to look for the source. The shifted arrays are then summed along the hydrophone axis. The range of possible angles is then given by taking the argmax of the squared sum of the result. Note that the thime domain beamforming does not have enough accuracy to give just one set of angles, intead returning an area of possible angles where the sound source might be.


## Fast Angle of Arrival

To get the angle of arrival of a signal source one can take the argmax of the value returned by any of the beamforming algorithms described above.

Although the frequency beamforming is accurate, it is slow due to the fact that it needs to compute the FFT of the entire signal for each angle where the signal source might be in and then multiply it by the FFT of the delays matrix (note that it uses the dot product). Although the time domain beamforming is faster than the frequency one, due to the fact that it only shifts the signal for each of the angles delays, it returns a wide range of values where the source might be.

To speed up the angle of arrival computation, while maintaing the same accuracy of the frequency domain bomain, the fast angle of arrival algorithm is proposed. It works by narrowing the list of possible angles by computing the time domain beamforming and then computing the frequency domaing beamforming on the given angles. The time domain computation also skips certain angles, as angles near one another generally have the same value. The amount of angles skiped is called determined by the letter "n".

The algorithm results are the same as the pure frequency domain beamforming approach, but with the speed being around 10 times faster.


## Speed Test

Running each of the angle of arrival techniques 100 times for the same signal source on a
Intel(R) Core(TM) i5-4440 CPU @ 3.10GHz the following times, in seconds, were given.

|      | Fast  | Frequency | Time  |
|------|-------|-----------|-------|
| mean | 0.028 |   1.289   | 0.460 |

As it can be seem, the fast angle of arrival is around 45 times faster than the pure frequency beamforming approach and 16 times faster than the pure time domain beamforming approach.

## Usage

A sample use of the fast angle of arival is shown bellow

```python
import beamforming as bf

distance = 3 * 10**-2 # Distance between sensors in m

# XY matrix of the sensors coordinates
coord = np.array((
                [0,0,distance],
                [0,0,0],
                [0,0,-distance],
                [-distance,0,0]
    ))

fs = 2e+6 # Signal sampling rate

amount_read = 256 # Signal size

# Create an object with the signal and sensor characteristics
b = bf.bf(coord, fs, amount_to_read)

signal = ... # Signal to read

azimuth, elevation = b.fast_aoa(signal)

print(azimuth, elevation)
```

To determine the best value of n(how many angles to skip in the time beamforming computation) to a given audio, run
```
python n_example.py
```
It will run a simulation with different values of n(up to 20) and plot a graph with their times. To determine which audio the simulation must be run in change the following line in n_examples.py: 

```python
y, fs = sf.read('wavs/110118_002.WAV')
```
to

```python
y, fs = sf.read('your_file_path.WAV')
```

The soubndfile python library must be used to run the simulation.
