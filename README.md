# Fast Angle of Arrival

The fast beamforming is a fast implementation of the frequency domain beamforming algorithm.

# Table of Contents
1. [Frequency Domain Beamforming](#Frequency-Domain-Beamforming)
2. [Time Domain Beamforming](#Time-Domain-Beamforming)
3. [Fast Angle of Arrival](#Fast-Angle-of-Arrival)
4. [Speed Test](#Speed-Test)
5. [Usage](#Usage)

## Frequency Domain Beamforming

The frequency domain beamforming algorithm returns the azimuth and elevation angles of a signal source given an array of sensors that capture the signal.

The frequency beamforming works by multiplying the fast fourier transform (FFT) of the signal given by each sensor with the FFT of the delays matrix associated with the sensors position array and the angles to look for the source, taking the inverse FFT of the result and summing along the hydrophones axis. The final azimuth and elevation angles is given by getting the argmax of the squared sum of result.


## Time Domain Beamforming

The time domain beamforming algorithm returns an array of possible azimuth and elevation angles of a signal source given an array of sensors that capture the signal.

The time beamforming works by padding the given sensor signals with zeros and shifting the signal in time by the delays matrix associated with the sensors position array and the angles to look for the source. The shifted arrays are then summed along the hydrophone axis. The range of possible angles is then given by taking the argmax of the squared sum of the result.


## Fast Angle of Arrival

To get the angle of arrival of a signal source one can take the argmax of the value returned by any of the beamforming algorithms described above.

Although the frequency beamforming is accuratem it is slow due to the fact that it needs to compute the FFT of the entire signal for each angle
where the signal source might and then multiply it by the FFT of the delays matrix. Although the time domain beamforming is faster than the
frequency one, due to the fact that it only shifts the signal for each of the angles delays, it returns a range of values where the source might be.

To speed up the angle of arrival computation, while maintaing the same accuracy of the frequency domain bomain, the fast angle of arrival algorithm is
proposed. It works by narrowing the list of possible angles by computing the time domain beamforming and then computing the frequency domaing beamforming
on the given angles.

The algorithm results are the same as the pure frequency domain beamforming approach, but 10 times faster speed.


## Speed Test

Running each of the angle of arrival techniques 100 times for the same signal source on a
Intel(R) Core(TM) i5-4440 CPU @ 3.10GHz the following times, in seconds, were given.

|      | Fast  | Frequency | Time  |
|------|-------|-----------|-------|
| mean | 0.028 |   1.289   | 0.460 |

As it can be seem, the fast angle of arrival is around 45 times faster than the pure frequency beamforming approach and 16 times faster than the
pure time domain beamforming approach.

## Usage

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