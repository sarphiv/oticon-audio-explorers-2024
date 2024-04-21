from dataclasses import dataclass
from pathlib import Path
from typing import cast

import tyro
from tqdm import tqdm
import torch as th
import numpy as np
import torchiva
from scipy.io import wavfile
from scipy import signal
from scipy.signal import resample
from speechbrain.processing.features import STFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import Music, doas2taus
from df import enhance, init_df
from asteroid.models import DPTNet


@dataclass(frozen=True)
class Args:
    input_dir: Path
    output_dir: Path



@dataclass(frozen=True)
class EnhancedAudio:
    """A dataclass containing the enhanced audio and their sample rates."""
    separated: tuple[np.ndarray, int]
    natural: tuple[np.ndarray, int]
    suppressed: tuple[np.ndarray, int]



CLEAN_SPEECH_TIME = 2.0

dfn_model = init_df("DeepFilterNet3", log_level="none")
dptnet_model = DPTNet.from_pretrained('JorisCos/DPTNet_Libri1Mix_enhsingle_16k').eval().cuda()
# NOTE: Hard coded to avoid dependency upon path
mic_pos = np.array([
    [-3.92103779e-02, -0.00000000e+00,  1.50514539e-02],
    [-3.56180200e-02, -2.22566091e-02,  2.57175828e-18],
    [-3.92103779e-02, -0.00000000e+00, -1.50514539e-02],
    [-3.56180200e-02,  2.22566091e-02,  2.57175828e-18],
    [-2.22566091e-02, -0.00000000e+00,  3.56180200e-02],
    [-2.43275745e-02, -2.43275745e-02,  2.40902103e-02],
    [-1.50514539e-02, -3.92103779e-02,  2.57175828e-18],
    [-2.43275745e-02, -2.43275745e-02, -2.40902103e-02],
    [-2.22566091e-02, -0.00000000e+00, -3.56180200e-02],
    [-2.43275745e-02,  2.43275745e-02, -2.40902103e-02],
    [-1.50514539e-02,  3.92103779e-02,  2.57175828e-18],
    [-2.43275745e-02,  2.43275745e-02,  2.40902103e-02],
    [ 2.62684091e-04, -1.50491615e-02,  3.92103779e-02],
    [-2.18097471e-18, -3.56180200e-02,  2.22566091e-02],
    [-2.20442710e-18, -3.60010266e-02, -2.16315991e-02],
    [-2.62684091e-04, -1.50491615e-02, -3.92103779e-02],
    [ 3.92103779e-02, -4.80188638e-18,  1.50514539e-02],
    [ 3.56180200e-02,  2.22566091e-02,  2.57175828e-18],
    [ 3.92103779e-02, -4.80188638e-18, -1.50514539e-02],
    [ 3.56180200e-02, -2.22566091e-02,  2.57175828e-18],
    [ 2.22566091e-02, -2.72564851e-18,  3.56180200e-02],
    [ 2.43275745e-02,  2.43275745e-02,  2.40902103e-02],
    [ 1.50514539e-02,  3.92103779e-02,  2.57175828e-18],
    [ 2.43275745e-02,  2.43275745e-02, -2.40902103e-02],
    [ 2.22566091e-02, -2.72564851e-18, -3.56180200e-02],
    [ 2.43275745e-02, -2.43275745e-02, -2.40902103e-02],
    [ 1.50514539e-02, -3.92103779e-02,  2.57175828e-18],
    [ 2.43275745e-02, -2.43275745e-02,  2.40902103e-02],
    [ 2.62684091e-04,  1.50491615e-02,  3.92103779e-02],
    [ 6.54292413e-18,  3.56180200e-02,  2.22566091e-02],
    [ 6.54292413e-18,  3.56180200e-02, -2.22566091e-02],
    [-2.62684091e-04,  1.50491615e-02, -3.92103779e-02]
])



def align_audio(audio_raw: np.ndarray, sample_rate: int, microphone_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Aligns the audio channels using MUSIC for DOA estimation.

    Args:
        audio_raw (np.ndarray): The raw audio data with shape (num_channels, num_samples).
        sample_rate (int): The sample rate of the audio.
        microphone_idx (np.ndarray | None): Indices of subset of microphones to use. Set to None to use all microphones.

    Returns:
        tuple[np.ndarray, np.ndarray]: Aligned audio data and directions of arrival
            - The aligned audio data with shape (num_channels, num_samples).
            - DIrections of arrival of audio (num_time_frames, 3).
    """
    # WARN: Very nasty and hacky competition code, look away


    # If microphone indices provided, only use those for alignment
    if microphone_idx is not None:
        audio_raw = audio_raw[microphone_idx]
    # Else, use all microphone
    else:
        microphone_idx = np.arange(audio_raw.shape[0])


    # Define the first few seconds to be clean speech
    clean_idx = int(CLEAN_SPEECH_TIME * sample_rate)

    # Estimate DOA with MUSIC
    covariance = Covariance().cuda()(STFT(
        sample_rate=sample_rate, 
        n_fft=512, 
        hop_length=160*1000//sample_rate, 
        win_length=512*1000//sample_rate
    ).cuda()(
        th.tensor(audio_raw[:, :clean_idx], dtype=th.float32).T.view(1, clean_idx, audio_raw.shape[0]).cuda()
    ))

    music = Music(mics=th.tensor(mic_pos[microphone_idx], dtype=th.float32).cuda(), sample_rate=sample_rate).cuda()
    doas: th.Tensor = music(covariance)[0]

    # Get time deltas
    taus: np.ndarray = doas2taus(
        doas=doas.view(1, *doas.shape), 
        mics=th.tensor(mic_pos[microphone_idx], dtype=th.float32), 
        fs=sample_rate
    )[0].numpy().astype(np.int64)

    # Align channels
    audio_aligned = np.zeros(audio_raw.shape)
    lags: np.ndarray = taus.mean(0).astype(np.int64)

    for i in range(audio_raw.shape[0]):
        right_pad = max(0, -lags[i])
        left_pad = max(0, lags[i])
        if right_pad > 0:
            audio_aligned[i] = np.pad(audio_raw[i], (0, right_pad))[right_pad:]
        elif left_pad > 0:
            audio_aligned[i] = np.pad(audio_raw[i], (left_pad, 0))[:-left_pad]
        else:
            audio_aligned[i] = audio_raw[i]

    # Return results
    return audio_aligned, doas.cpu().numpy()



def choose_microphones(doas: np.ndarray) -> np.ndarray:
    """Choose the four microphones closest to the direction of arrival. Assuming stationary speaker.
    
    Args:
        doas (np.ndarray): The direction of arrival estimates with shape (num_time_frames, 3).

    Returns:
        np.ndarray: The indices of the four microphones closest to the direction of arrival.
    """
    # Get unit direction
    direction = doas.mean(0)
    direction /= np.linalg.norm(direction, 2)
    
    # Normalize microphone positions to unit vectors
    mic_pos_unit = mic_pos / np.linalg.norm(mic_pos, 2, axis=1)[:, None]


    # Get indices of four closest microphones
    return np.sort(np.argpartition(np.linalg.norm(mic_pos_unit - direction, 2, axis=1), 4)[:4])


def separate_audio(audio_aligned: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Separates audio sources using Independent Vector Analysis (AuxIVA-IP) algorithm.

    Args:
        audio_aligned (np.ndarray): The aligned audio data as a numpy array with shape (num_channels, num_samples)
        sample_rate (int): The sample rate of the audio data.

    Returns:
        np.ndarray: The separated audio sources as a numpy array.
    """
    
    # Declare STFT
    window = signal.windows.hann(256)
    stft = signal.ShortTimeFFT(window, 160, sample_rate)

    # Perform IVA
    audio_stft = stft.stft(audio_aligned, axis=1)
    iva_model = torchiva.AuxIVA_IP(
        n_iter=100, 
        n_src=4, 
        model=torchiva.models.LaplaceModel()
    )
    sources_stft = iva_model.forward(th.tensor(audio_stft).cuda())

    # Perform inverse STFT
    return stft.istft(sources_stft.cpu().numpy())



def choose_best_source(sources: np.ndarray, audio_aligned: np.ndarray, sample_rate: int) -> int:
    """
    Chooses the best source from the separated sources.

    Args:
        sources (np.ndarray): The separated audio sources as a numpy array with shape (num_sources, num_samples)

    Returns:
        int: The index of the best source.
    """
    # Define the first few seconds to be clean speech
    clean_idx = int(CLEAN_SPEECH_TIME * sample_rate)

    # Measure similarity to original audio and return the index of the most similar source
    importance = np.abs(audio_aligned[:, :clean_idx] @ sources[:, :clean_idx].T).sum(0)
    return cast(int, np.argmax(importance))


@th.no_grad()
def enhance_audio(audio_raw: np.ndarray, sample_rate: int, microphone_idx: np.ndarray | None = None) -> EnhancedAudio:
    """
    Enhances the input audio by performing various processing steps.

    Args:
        audio_raw (np.ndarray): The raw audio data as a numpy array with shape (num_channels, num_samples).
        sample_rate (int): The sample rate of the audio.
        microphone_idx (np.ndarray | None): Indices of subset of microphones to use (num_mics,). Set to None to use all microphones.

    Returns:
        EnhancedAudio: A dataclass containing the enhanced speech at various stages with their sample rates:
            - The separated speech source with shape (num_samples,).
            - The enhanced and natural sounding speech with shape (num_samples,).
            - The enhanced and heavily noise suppressed speech (num_samples,).
    """
    # Normalize the audio
    audio_raw = audio_raw.astype(np.float32)
    audio_raw = (audio_raw - audio_raw.mean(axis=1, keepdims=True)) / np.abs(audio_raw).max(axis=1, keepdims=True)

    # Align audio before IVA
    audio_aligned, doas = align_audio(audio_raw, sample_rate, microphone_idx)

    # Choose closest microphones
    mic_idx = choose_microphones(doas)

    # Separate into sources
    sources = separate_audio(audio_aligned[mic_idx], sample_rate)

    # Choose best source
    speech_idx = choose_best_source(sources, audio_aligned, sample_rate)

    # Enhance with DeepFilterNet 3
    # NOTE: Magic multiplication with the number 3 as it sounds more pleasant with DPTNet
    audio_natural = enhance(
        *dfn_model[:2],
        th.tensor(sources[None, speech_idx] * 3, dtype=th.float32),
        atten_lim_db=8
    ).numpy()

    # Enhance with DPTNet
    audio_natural_resampled = th.tensor(
        resample(audio_natural, int(audio_natural.shape[1]/sample_rate * dptnet_model.sample_rate), axis=1),
        dtype=th.float32
    )
    audio_suppressed = dptnet_model(audio_natural_resampled.cuda())[0, 0].cpu().numpy()


    # Return enhanced speech at various stages
    normalize = lambda x: np.array(x.squeeze() / np.abs(x).max(), dtype=np.float32)

    return EnhancedAudio(
        separated=(normalize(sources[speech_idx]), sample_rate),
        natural=(normalize(audio_natural), sample_rate),
        suppressed=(normalize(audio_suppressed), int(dptnet_model.sample_rate))
    )




if __name__ == "__main__":
    # Load arguments
    args = tyro.cli(Args)


    for file in tqdm(args.input_dir.glob("*.wav")):
        # Load audio
        sample_rate, audio_raw = wavfile.read(file)
        audio_enhanced = enhance_audio(audio_raw.T, sample_rate)


        # Save enhanced audio
        args.output_dir.mkdir(exist_ok=True)
        save_path_prefix = str(args.output_dir / f"{file.stem}")

        wavfile.write(
            save_path_prefix + "-separated.wav",
            audio_enhanced.separated[1], audio_enhanced.separated[0]
        )
        
        wavfile.write(
            save_path_prefix + "-natural.wav",
            audio_enhanced.natural[1], audio_enhanced.natural[0]
        )
        
        wavfile.write(
            save_path_prefix + "-suppressed.wav",
            audio_enhanced.suppressed[1], audio_enhanced.suppressed[0]
        )
