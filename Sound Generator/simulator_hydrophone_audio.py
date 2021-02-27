import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import beamforming as bf
import sys

class plot():
    def __init__(self):
        plt.axis([0,6137332,-0.5,0.5])
    
    def show(self,xs,ys):
        plt.plot(xs,ys)
        plt.show()
        plt.close()

class audio_simulator():
    def __init__(self,sample_audio=None,coords=None,fs=None,setup=True):
        if setup:
            self.simulatorSetup()
        else:
            self.sample_audio = np.array([[x]*len(coords) for x in sample_audio[...,0]])
            self.coords = coords
            self.fs = fs
    
    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    
    def shift_array(self,array,shift):
        a = array[len(array)-shift:]
        b = array[0:shift]
        c = array[shift:len(array)-shift]
        return np.concatenate([a,b,c])
    
    def beamformAt(self,x,y,z):
        # Make audio
        audio = self.sample_audio
        #sound_speed  = 1491.24 # m/s
        sound_speed = 1450 # m/s
        pinger = np.array([x,y,z])
        displacements = []
        for i in self.coords:
            i = pinger - i
            i = sum(i**2)**0.5
            i *= self.fs/sound_speed
            displacements.append(int(i))
        for j in range(len(displacements)):
            audio[...,j] = self.shift_array(audio[...,j],displacements[j])
        
        # Get true azimuth and elevation
        azi_aux,ele_aux = [x,y,0],[x,y,z]
        z_axis,y_axis = [0,0,1],[0,-1,0]
        azimuth = self.angle_between(azi_aux,y_axis)
        elevation = self.angle_between(ele_aux, z_axis)

        return audio,self.fs,azimuth,elevation
    
    def simulatorSetup(self):
        distance_x = (19.051e-3)/2
        distance_y = (18.37e-3)/2
        coords = np.array([
        [-distance_x, -8.41e-3, -distance_y],
        [distance_x, 0, -distance_y],
        [distance_x, -8.64e-3, distance_y],
        [-distance_x, -0.07e-3, distance_y]
        ])
        sample_audio, fs = sf.read('030719_013.WAV')
        self.sample_audio = np.array([[x]*len(coords) for x in sample_audio[...,0]])
        self.coords = coords
        self.fs = fs

if __name__ == "__main__":
    #p = plot()

    sim = audio_simulator()
    audio,fs,az,el = sim.beamformAt(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]))
    print(az - 90,180 - el)
    sf.write('generated_audio.WAV',audio,fs)

    # Plot Audio Representation of Channel 2
    #ys = audio[...,1]
    #xs = [x for x in range(len(ys))]
    #p.show(xs,ys)
