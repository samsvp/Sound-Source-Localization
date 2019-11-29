# Fast Beamforming

The fast beamforming is a fast implementation of the frequency domain beamforming algorithm.

## Frequency Domain Beamforming

The frequency domain beamforming algorithm returns the azimuth and elevation angles of a signal source given an array of sensors that capture the signal.

The frequency beamforming works by multiplying the fast fourier transform (FFT) of the signal given by each sensor with the FFT of the delays matrix associated with the sensors position array and the angles to look for the source, taking the inverse FFT of the result and summing along the hydrophones axis. The final azimuth and elevation angles is given by getting the argmax of the squared sum of result.

## Time Domain Beamforming

The time domain beamforming algorithm returns an array of possible azimuth and elevation angles of a signal source given an array of sensors that capture the signal.

The time beamforming works by padding the given sensor signals with zeros and shifting the signal in time by the delays matrix associated with the sensors position array and the angles to look for the source. The shifted arrays are then summed along the hydrophone axis. The range of possible angles is then given by taking the argmax of the squared sum of the result.


## Fast Beamforming

Although the frequency beamforming is accuratem it is slow due to the fact that it needs to compute the FFT of the entire signal for each angle where the sound source might and then multiply it by the FFT of the delays matrix. Although the time domain beamforming is faster than the frequency one, due to the fact that it only shifts the signal for each of the angles delays, it returns a range of values where the source might be.

To speed up the beamforming algorithm while maintaing the same accuracy of the frequency domain bomain, the fast beamforming algorithm is proposed. It works by narrowing the list of possible angles by computing the time domain beamforming and then computing the frequency domaing beamforming on the given angles.

The algorithm results are the same as the frequency domain beamforming but at more than half the speed.

## Speed test

Running each of the beamforming algorithms on a Intel(R) Core(TM) i3-5020U CPU @ 2.20GHz the following times where given.


