from pystoi import stoi
from pesq import pesq
import numpy as np
import os 
from modelling.enhance import enhance_audio
from pathlib import Path
from scipy.io import wavfile
import tqdm
import scipy
from scipy.signal import resample
import librosa
import fast_align_audio 


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.max(np.abs(x))


def calculate_snr(clean_audio: np.ndarray, enhanced_audio: np.ndarray) -> float:
    # Calculate the Signal to Noise Ratio (SNR) for a single channel
    # SNR = 10 * log10(P_signal / P_noise)
    
    signal_power = np.sum(clean_audio**2)
    noise_power = np.sum((clean_audio - enhanced_audio)**2)

    return 10 * np.log10(signal_power / (noise_power + 1e-10))


def calculate_pesq(clean_audio, enhanced_audio, sample_rate) -> float:
    sample_rate_new = 16000
    clean_audio = resample(clean_audio, int(len(clean_audio) / sample_rate * sample_rate_new), axis=0)
    enhanced_audio = resample(enhanced_audio, int(len(enhanced_audio) / sample_rate * sample_rate_new), axis=0)

    return pesq(sample_rate_new, clean_audio, enhanced_audio, 'wb')


def calculate_estoi(clean_audio, enhanced_audio, sample_rate: int) -> float:
    return stoi(clean_audio, enhanced_audio, sample_rate, extended=True)


def align_audio(reference_audio: np.ndarray, desynced_audio: np.ndarray) -> np.ndarray:
    offset, _ = fast_align_audio.find_best_alignment_offset(
        reference_signal=reference_audio,
        delayed_signal=desynced_audio,
        max_offset_samples=20,
        lookahead_samples=60,
    )


    if offset < 0:
        return np.pad(desynced_audio[:offset], (-offset, 0))
    elif offset > 0:
        return np.pad(desynced_audio[offset:], (0, offset))
    else:
        return desynced_audio



def calculate_metrics(clean_audio: np.ndarray, clean_sr: int, enhanced_audio: np.ndarray, enhanced_sr: int) -> np.ndarray:
    # Normalize just in case
    clean_audio = normalize(clean_audio)
    enhanced_audio = normalize(enhanced_audio)

    # If the sampling rates are different, upsample the lower one
    if clean_sr != enhanced_sr:
        if clean_sr < enhanced_sr:
            clean_audio = resample(clean_audio, int(clean_audio.shape[1] / clean_sr * enhanced_sr), axis=1)
            clean_sr = enhanced_sr
        else:
            enhanced_audio = resample(enhanced_audio, int(enhanced_audio.shape[0] / enhanced_sr * clean_sr), axis=0)
            enhanced_sr = clean_sr


    # Sample rates are the same now, reflect that in the variable name
    sample_rate = enhanced_sr

    # Cut end of audio to have correct shape
    # NOTE: Noise sometimes continues after the speech ends, so cut the audio to the length of the clean audio
    # NOTE: Enhanced audio is not guaranteed to have same length because of inaccuracies in transformations
    length = min(clean_audio.shape[1], enhanced_audio.shape[0])
    clean_audio = clean_audio[:, :length]
    enhanced_audio = enhanced_audio[:length]


    # Align signals
    # TODO: Ensure the alignment lags aren't too crazy
    enhanced_audio = align_audio(clean_audio[0], enhanced_audio)

    for i in range(1, clean_audio.shape[0]-1):
        clean_audio[i] = align_audio(clean_audio[0], clean_audio[i])


    # Collapse audio to mono to enable metrics calculation
    clean_audio = clean_audio.mean(axis=0)

    
    return np.array([
        calculate_snr(clean_audio, enhanced_audio), 
        calculate_pesq(clean_audio, enhanced_audio, sample_rate), 
        calculate_estoi(clean_audio, enhanced_audio, sample_rate)
    ])



def run_metrics(clean_dir: Path, mixes_files: Path, mic_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean_files = list(clean_dir.glob("*.wav"))
    mixed_files = list(mixes_files.glob("*.wav"))

    # NOTE: Stored as: SNR, PESQ, STOI
    stats_sep = np.empty((len(mixed_files), 3))
    stats_nat = np.empty((len(mixed_files), 3))
    stats_sup = np.empty((len(mixed_files), 3))
    stats_raw = np.empty((len(mixed_files), 3))


    for i, (clean_file, mixed_file) in tqdm.tqdm(enumerate(zip(clean_files, mixed_files)), total=len(clean_files)):
        # Load and prepare the audio files
        clean_sr, clean_audio = wavfile.read(clean_file)
        mixed_sr, mixed_audio = wavfile.read(mixed_file)

        clean_audio, mixed_audio = normalize(clean_audio.T[mic_idx]), normalize(mixed_audio.T[mic_idx])

        # Run the enhancement
        enhanced_audio = enhance_audio(mixed_audio, mixed_sr, mic_idx)

        # Calculate metrics
        stats_sep[i] = calculate_metrics(clean_audio, clean_sr, *enhanced_audio.separated)
        stats_nat[i] = calculate_metrics(clean_audio, clean_sr, *enhanced_audio.natural)
        stats_sup[i] = calculate_metrics(clean_audio, clean_sr, *enhanced_audio.suppressed)
        stats_raw[i] = calculate_metrics(clean_audio, clean_sr, mixed_audio.mean(0), mixed_sr)


    # Return stats
    return stats_sep, stats_nat, stats_sup, stats_raw


if __name__ == "__main__":
    data_sets = ["sim_final","sim_final2","sim_final3"]
    for data_set in data_sets:
        clean_files = Path(f"data/{data_set}/clean")
        mixed_files = Path(f"data/{data_set}/mixed")
        mic_idx = np.array([0, 1, 2, 3])

        mic_arrays = np.array([
            [0, 1, 2, 3],
            [10, 12, 6, 8],
            [4, 1, 2, 17],
            [7, 14, 11, 30],
            [23, 11, 27, 7],
            [4, 2, 18, 20],
            [8, 10, 22, 28],
            [9, 5, 2, 4],
            [4, 8, 20, 22]
        ])

        outstr = ""
        for i, mic_idx in enumerate(mic_arrays):
            stats_sep, stats_nat, stats_sup, stats_raw = run_metrics(clean_files, mixed_files, mic_idx)
            
            stats_sep = np.round(stats_sep.mean(0), 3)
            stats_nat = np.round(stats_nat.mean(0), 3)
            stats_sup = np.round(stats_sup.mean(0), 3)
            stats_raw = np.round(stats_raw.mean(0), 3)
            
            print("intermediate results:")
            print(stats_sep)
            print(stats_nat)
            print(stats_sup)
            print(stats_raw)
            
            for idx in mic_idx:
                outstr += f"{idx}, "
            outstr = outstr[:-2] + " & "
            outstr += str(stats_sep[0]) + " & " + str(stats_nat[0]) + " & " + str(stats_sup[0]) + " & " + str(stats_raw[0]) + " & "
            outstr += str(stats_sep[1]) + " & " + str(stats_nat[1]) + " & " + str(stats_sup[1]) + " & " + str(stats_raw[1]) + " & "
            outstr += str(stats_sep[2]) + " & " + str(stats_nat[2]) + " & " + str(stats_sup[2]) + " & " + str(stats_raw[2]) + " \\\\ \n"
            
            if (i+1)%2 == 0:
                outstr += "\\rowcolor{gray!20}\\cellcolor{white}\n"
            
        # save the string to a text file
        with open(f"{data_set}.txt", "w") as text_file:
            text_file.write(outstr)
            
        print(outstr)

    # print(" ==== AVERAGE METRICS ==== ")
    # print("Signal type |SNR |PESQ |ESTOI |")
    # print(f"Seperated:  {np.round(stats_sep.mean(0), 3)}")
    # print(f"Natural:    {np.round(stats_nat.mean(0), 3)}")
    # print(f"Suppressed: {np.round(stats_sup.mean(0), 3)}")
    # print(f"Raw:        {np.round(stats_raw.mean(0), 3)}")

#legendary'nt uncommon common rare epic legendary legendary'nt'nt