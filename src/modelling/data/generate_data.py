
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import scipy

import json

import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import scipy.io

import tqdm
import random

from dataclasses import dataclass

MIC_COORDS = Path("data/mic_pos.npy")

class RoomSimulator:
    
    def __init__(self, room_size, rverb_max = 0.05, rverb_min = 0.2, sr = 44100, mic_radius = 0.042, room_type = "dry"):
        
        self.rverb_max = rverb_max
        self.rverb_min = rverb_min
        
        self.sr = sr
        self.mic_radius = mic_radius
        
        self.room = self.get_room(room_size, room_type = room_type)
        
        self.mic_center = None
        self.target_signal = None
        self.target_position = None
        
        self.source_meta = []
        
    def get_mic_coords(self, center : np.ndarray, channels : np.ndarray):
        
        coords = np.load(MIC_COORDS)
        coords = coords[channels,:]/np.linalg.norm(coords[channels,:],axis=1)[...,None]
        coords *= self.mic_radius
        coords += center
        self.mic_center = center
        
        return coords
        
    def finalize_microphones(self, coords : np.ndarray):
        
        if self.mic_center is None:
            raise ValueError("Microphone array not initialized. Initialize microphones with get_mic_coords before finalizing")
        
        if self.target_position is None:
            raise ValueError("Target source not initialized. Initialize target source before finalizing")
        
        # Rorate the coordiantes so they are always facing channel [1,2,3,4]
        forward = np.array([-1,0,0])
        rotation_axis = np.cross(forward, self.target_position - self.mic_center)
        rotation_axis /= np.linalg.norm(rotation_axis)
        
        rotation_angle = np.arccos(np.dot(forward, (self.target_position - self.mic_center) / np.linalg.norm(self.target_position - self.mic_center)))
    
        r = scipy.spatial.transform.Rotation.from_rotvec(rotation_angle * rotation_axis)
         
        coords = r.apply(coords - self.mic_center[None,...]) + self.mic_center[None,...]
        
        
        colatitudes, azimuths = self.get_angles_from_coords(coords, origin = self.mic_center)
        self.mic_angles = np.vstack([colatitudes, azimuths])
        
        dir_objs = [CardioidFamily(
                        orientation=DirectionVector(azimuth=azimuths[i], colatitude=colatitudes[i], degrees = False),
                        pattern_enum=DirectivityPattern.CARDIOID)
                    for i in range(coords.shape[0])]
        
        self.mic_array = pra.Beamformer(coords.T, self.room.fs)
        
        self.room.add_microphone_array(self.mic_array, directivity = dir_objs)
        
        
    def get_room(self, room_size : np.ndarray = np.array([4,6,3]), room_type = "dry") -> pra.Room:
        self.room_size = room_size
        
        if room_type == "wet":
            rverb = np.random.uniform(self.rverb_min, self.rverb_max)
            room = pra.ShoeBox(self.room_size, fs=self.sr, max_order=1,
                                    use_rand_ism = True, max_rand_disp = 0.01,
                                    air_absorption=True,
                                    materials = pra.Material(rverb)
                                )
        elif room_type == "dry":
            room = pra.ShoeBox(self.room_size, fs=self.sr, max_order=0,
                                    use_rand_ism = True, max_rand_disp = 0.01,
                                    air_absorption=True)
        else:
            raise ValueError("Invalid room type: Should be 'dry' or 'wet'")
            
        return room
    
    def add_target_source(self, position, signal):
        """
        Add a voice source to the room that points towards the microphone array
        """

        if self.mic_center is None:
            raise ValueError("Microphone array not initialized. Initialize microphones before audio")
        
        colatitude, azimuth = self.get_angles_from_coords(self.mic_center[None,...], origin = position)
        
        directivity = CardioidFamily(
                        orientation=DirectionVector(azimuth=azimuth, colatitude=colatitude, degrees = False),
                        pattern_enum=DirectivityPattern.CARDIOID)

        self.room.add_source(position, signal = signal, directivity = directivity)
        
        self.target_position = position
        self.target_source = signal
        
        self.source_meta.append({"type": "target", "position" : list(position), "colatitude" : colatitude[0], "azimuth" : azimuth[0]})

    
    def get_angles_from_coords(self, coords : np.ndarray, origin : np.ndarray = np.array([0,0,0])) -> np.ndarray:
        """
        Given a Nx3 array of coordinates and a 3x1 origin point,
        return the colatitude and azimuth angles of the coordinates
        """
        
        coords_ = coords - origin
        
        colatitudes = np.arccos(coords_[:,2]/np.sqrt(coords_[:,0]**2 + coords_[:,1]**2 + coords_[:,2]**2))
        azimuths = np.zeros(coords_.shape[0])
        for i in range(coords_.shape[0]):
            if coords_[i,0] > 0:
                azimuths[i] = np.arctan(coords_[i,1]/coords_[i,0])
            elif coords_[i,0] < 0 and coords_[i,1] >= 0:
                azimuths[i] = np.arctan(coords_[i,1]/coords_[i,0]) + np.pi
            elif coords_[i,0] < 0 and coords_[i,1] < 0:
                azimuths[i] = np.arctan(coords_[i,1]/coords_[i,0]) - np.pi
            elif coords_[i,0] == 0 and coords_[i,1] > 0:
                azimuths[i] = np.pi/2
            elif coords_[i,0] == 0 and coords_[i,1] < 0:
                azimuths[i] = -np.pi/2
            elif coords_[i,0] == 0 and coords_[i,1] == 0:
                raise ValueError("Mic position is at the origin")
            
        return np.vstack([colatitudes, azimuths])
    
    def get_coords_from_angles(self, colatitudes : np.ndarray, azimuths : np.ndarray, origin : np.ndarray = np.array([0,0,0])) -> np.ndarray:
        """
        Given a Nx1 array of colatitudes and azimuths, return the coordinates
        """
        
        x = np.sin(colatitudes) * np.cos(azimuths)
        y = np.sin(colatitudes) * np.sin(azimuths)
        z = np.cos(colatitudes)
        
        coords = np.vstack([x, y, z]).T
        coords += origin
        
        return coords.T
        
    def plot_room(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        self.room.plot(ax=ax, plot_directivity=False)
        
        # plot the microphones
        ax.scatter(self.mic_array.R[0,:], self.mic_array.R[1,:], self.mic_array.R[2,:], c='r', marker='o', s=11)
        
        colatitudes, azimuths = self.mic_angles
        x, y, z = self.get_coords_from_angles(colatitudes, azimuths)

        ax.quiver(self.mic_array.R[0,:], self.mic_array.R[1,:], self.mic_array.R[2,:], x, y, z, length=1)
        
        # plot text on the microphones at the end of the quiver
        quiver_length = 1.5
        for i in range(self.mic_array.R.shape[1]):
            ax.text(self.mic_array.R[0,i] + x[i]*quiver_length, self.mic_array.R[1,i] + y[i]*quiver_length, self.mic_array.R[2,i] + z[i]*quiver_length, str(i))
        
        # plot the sources
        for source in self.source_meta:
            if "colatitude" in source.keys():
                x, y, z = self.get_coords_from_angles(source["colatitude"], source["azimuth"])
                ax.quiver(source["position"][0], source["position"][1], source["position"][2], x, y, z, length=quiver_length, color = 'r')
            
        
        # set limits to self.room_size
        ax.set_xlim([0, self.room_size[0]])
        ax.set_ylim([0, self.room_size[1]])
        ax.set_zlim([0, self.room_size[2]])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.show()
        
    def simulate_room(self) -> np.ndarray:
        
        self.room.compute_rir()
        self.room.simulate()
        
        return self.room.mic_array.signals # type:ignore
    
    def _get_random_placement(self, wall_dist, mic_dist = None, try_iter = 100):
        
        for i in range(try_iter):
            pos_x = np.random.uniform(wall_dist[0], self.room_size[0] - wall_dist[0])
            pos_y = np.random.uniform(wall_dist[1], self.room_size[1] - wall_dist[1])
            pos_z = np.random.uniform(wall_dist[2], self.room_size[2] - wall_dist[2])
            position = np.array([pos_x, pos_y, pos_z])
            
            if mic_dist is not None:
                if np.linalg.norm(position - self.mic_center) > mic_dist:
                    break
            else:
                break
            
            if i == try_iter - 1:
                raise ValueError("Could not find a suitable location for white noise source")
        
        return position
    
    def add_room_noise(self, length, whtie_loc = 0, white_scale = 1, wall_dist = [1,1,0.5]):
        """
        Add a source of white and a source of pink noise to the room.
        Place the sources at random locations but at least 2 meters away from the microphone array 
        and 1 meter away from the walls.
        """
        
        white_noise = np.random.normal(loc = whtie_loc, scale = white_scale, size = length)
        
        position = self._get_random_placement(wall_dist = wall_dist, mic_dist = 2)
            
        self.room.add_source(position, signal = white_noise)
        
        self.source_meta.append({"type": "room_noise", "position" : list(position)})
        
    def snr_scale_factor(self, noise: np.ndarray, snr: int):
        """
        Compute the scale factor that has to be applied to a noise signal in order for the noisy (sum of noise and clean)
        to have the specified SNR.

        :param noise: the noise signal [..., SAMPLES]
        :param snr: the SNR of the mixture
        :return: the scaling factor
        """
        if self.target_source is None:
            raise ValueError("Target signal not initialized. Initialize target signal before noise")

        noise_var = np.mean(np.var(noise, axis=-1))
        speech_var = np.mean(np.var(self.target_source, axis=-1))

        factor = np.sqrt(
            speech_var / np.maximum((noise_var * 10. ** (snr / 10.)), 10**(-6)))

        return factor
        
    def add_noise_signal(self, signal, position, delay, snr):
        """
        Add a noise signal to the room at a certain position with a certain SNR
        """
        
        noise_scale = self.snr_scale_factor(signal, snr)
        
        scaled_noise = signal * noise_scale
        
        if self.mic_center is None:
            raise ValueError("Microphone array not initialized. Initialize microphones before audio")

        self.room.add_source(position, signal = scaled_noise, delay = delay)
        
        self.source_meta.append({"type": "noise", "position" : list(position), "snr" : snr, "delay" : delay})
            
    
    
class Config:
    
    seed = 43
    
    n_noise_signals_min: int = 4
    n_noise_signals_max: int = 5
    mic_radius: float = 0.042
    room_max_dim: np.ndarray = np.array([10,10,5])
    room_min_dim: np.ndarray = np.array([6,6,3])
    snr_min : int = 1
    snr_max : int = 10
    
    sr : int = 44100
    
    target_wall_min_dist : list = [1, 1, 0.5]
    target_mic_min_dist : float = 0.5
    
    mic_wall_min_dist : list = [2, 2, 0.5]
    channels : np.ndarray = np.arange(32)
    
    voice_data_path : Path = Path("data/voice_data_splits/train")
    noise_data_path : Path = Path("data/noice/train")
    
    noise_delay : float = 2.2
    
class DataCreator():
    
    def __init__(self, n_samples : int, out_dir : Path, config : Config):
        
        self.config = config
        self.out_dir = out_dir
        self.n_samples = n_samples
        
        self._seed_everything()
        self._create_dirs()
        
    def _create_dirs(self):
            
        for dir_ in ["clean", "mixed", "meta"]:
            dir_path = self.out_dir / dir_
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def _seed_everything(self):
        
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
    def create_all_samples(self, plot = False):
        
        voice_paths = np.array(list(self.config.voice_data_path.glob("*.wav")))
        noise_paths = np.array(list(self.config.noise_data_path.glob("*.wav")))
        
        np.random.shuffle(voice_paths)
        np.random.shuffle(noise_paths)
        
        print("STATUS: Found ", len(voice_paths), " voice samples")
        print("STATUS: Found ", len(noise_paths), " noise samples")
        
        if self.n_samples > len(voice_paths):
            print("NOTE: Not enough voice samples. Only creating: ", len(voice_paths))
            self.n_samples = len(voice_paths)
        
        for i in tqdm.trange(self.n_samples):
            n_noise_samples = np.random.randint(self.config.n_noise_signals_min,
                                                self.config.n_noise_signals_max + 1)
            
            noise_samples = np.random.choice(noise_paths, n_noise_samples)
            voice_sample = voice_paths[i]
            
            clean, mixed, meta = self.create_sample(voice_sample, noise_samples, plot)

            # Normalize the signals 
            clean = clean / np.max(np.abs(clean))
            mixed = mixed / np.max(np.abs(mixed))
            
            # convert to from 32bitfloating point to 32bit integer
            clean = (clean * (2**23 - 1)).astype(np.int32)
            mixed = (mixed * (2**23 - 1)).astype(np.int32)
            
            # Save the samples as .wav files
            scipy.io.wavfile.write(self.out_dir / "clean" / f"{i}.wav", self.config.sr, clean.T)
            scipy.io.wavfile.write(self.out_dir / "mixed" / f"{i}.wav", self.config.sr, mixed.T)
            
            # Save the metadata as json file
            with open(self.out_dir / "meta" / f"{i}.json", "w") as f:
                json.dump({i : m for m in meta}, f)
            
        
    def create_sample(self, voice_sample : Path, noise_samples : np.ndarray, plot = False):
        
        sims = self._get_rooms()
        clean_sim : RoomSimulator = sims[0]
        mix_sim : RoomSimulator = sims[1]
        coords : np.ndarray = sims[2]
        
        # Set the target source
        target_pos = clean_sim._get_random_placement(wall_dist = self.config.target_wall_min_dist,
                                               mic_dist = self.config.target_mic_min_dist)
        sr, voice_signal = scipy.io.wavfile.read(voice_sample)
        clean_sim.add_target_source(target_pos, voice_signal)
        mix_sim.add_target_source(target_pos, voice_signal)
        
        clean_sim.finalize_microphones(coords)
        mix_sim.finalize_microphones(coords)
        
        # Add noise signals
        for noise_sample in noise_samples:
            snr = np.random.randint(self.config.snr_min, self.config.snr_max)
            
            sr, noise_signal = scipy.io.wavfile.read(noise_sample)
            noise_pos = mix_sim._get_random_placement(wall_dist = self.config.mic_wall_min_dist)
            mix_sim.add_noise_signal(noise_signal,
                                     noise_pos,
                                     delay = self.config.noise_delay,
                                     snr = snr)
        
        if plot:
            mix_sim.plot_room()
        
        # Add room noise
        mix_sim.add_room_noise(length = voice_signal.shape[0])
        
        clean_signals = clean_sim.simulate_room()
        mix_signals = mix_sim.simulate_room()
        
        return clean_signals, mix_signals, mix_sim.source_meta
        
    def _get_rooms(self) -> tuple:
        
        room_x = np.random.uniform(self.config.room_min_dim[0], self.config.room_max_dim[0])
        room_y = np.random.uniform(self.config.room_min_dim[1], self.config.room_max_dim[1])
        room_z = np.random.uniform(self.config.room_min_dim[2], self.config.room_max_dim[2])
        room_dim = np.array([room_x, room_y, room_z])
        
        clean_sim = RoomSimulator(room_dim, mic_radius=self.config.mic_radius, room_type="dry", sr = self.config.sr)
        mix_sim = RoomSimulator(room_dim, mic_radius=self.config.mic_radius, room_type="wet", sr = self.config.sr)
        mic_pos = clean_sim._get_random_placement(wall_dist = self.config.mic_wall_min_dist)
        
        coords = clean_sim.get_mic_coords(mic_pos, self.config.channels)
        coords = mix_sim.get_mic_coords(mic_pos, self.config.channels)
        
        return clean_sim, mix_sim, coords
        

if __name__ == "__main__":
    
    dc = DataCreator(10, Path("sim_data_lesser_noise"), Config())
    dc.create_all_samples(plot = False)
    