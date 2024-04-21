from pystoi import stoi
from pesq import pesq
import numpy as np
import os 
import librosa
from modelling.enhance import enhance_audio


def calculate_snr(reference_signal, target_signal):
    # Calculate the Signal to Noise Ratio (SNR) for a single channel
    # SNR = 10 * log10(P_signal / P_noise)
    signal_power = np.mean(target_signal ** 2)
    noise_power = np.mean((target_signal - reference_signal) ** 2)
    if noise_power == 0:
        return float('inf')  # Avoid division by zero
    snr = 10 * np.log10(signal_power / noise_power)
    return snr




def calculate_metrics(true_signal, audio_arrays):

    """
    Input:
    true_signal: numpyarray
    audio_arrays: list of numpy arrays extracted from EnhancedAudio

    Output: 
    signal_dict:  dictionary with keys as indices and values as another dictionary of scores
    """

    signal_dict = {}
    ar_list = ["Seperated Audio", "Natural Audio", "Suppressed Audio"]


    # 16000 Hz aligns better with 'wb' mode in PESQ
    default_sr = 16000

    min_length = min(len(true_signal), *(len(x) for x in audio_arrays))
    true_signal = true_signal[:min_length]

    for i, enhanced_sig in enumerate(audio_arrays):
        # Trim all signals so they are the same size
        enhanced_sig = enhanced_sig[:min_length]

        metrics = {}
        
        # Compute PESQ score
        # pesq(fs, ref, deg, mode='wb', on_error=PesqError.RAISE_EXCEPTION)
        # fs:  integer, sampling rate
        # ref: numpy 1D array, reference audio signal
        # deg: numpy 1D array, degraded audio signal
        try:
            pesq_score = pesq(default_sr, true_signal, enhanced_sig, 'wb')  # 'wb' is for wide-band
            metrics['PESQ'] = pesq_score
        except Exception as e:
            print(f"PESQ: {e}")

        # Compute STOI score
        # stoi(x, y, fs_sig, extended=False)
        # x (np.ndarray): clean original speech
        # y (np.ndarray): denoised speech
        # fs_sig (int): sampling rate of x and y
        try:
            stoi_score = stoi(true_signal, enhanced_sig, default_sr, extended=False)
            metrics['STOI'] = stoi_score
        except Exception as e:
            print(f"STOI: {e}")
   
        # Compute the SNR score
        try:
            snr_score = calculate_snr(true_signal, enhanced_sig)
            metrics['SNR'] = snr_score
        except Exception as e:
            print(f"SNR: {e}")

        signal_dict[ar_list[i]] = metrics

    return signal_dict

#mic_idx feed
# np.array([1, 2, 3, 4])
# np.array([32, 12, 30, 14])


def run_metrics(clean_files, raw_files, mic_idx, sample_r = 16000):
    """
    Input:
    clean_files: (str) Directory with the clean files
    raw_files: (str) Directory with the raw files
    mic_idx: (np.array) A numpy array with 4 microphone indicies listed

    Output: 
    metrics_dict: (dict) A dictionary with mic_idx as keys and dicts as values with the metrics listed
    """
    # This is how I imagine the output to look like: {([1, 2, 3, 4]): {Sep: {PESQ: 1, STOI: 1, SNR: 1}, Natural: {PESQ: 1, STOI: 1, SNR: 1}, Supp: {PESQ: 1, STOI: 1, SNR: 1}}}

    metrics_dict = {}


    # List all files in directories
    clean_files_list = sorted([os.path.join(clean_files, f) for f in os.listdir(clean_files) if f.endswith('.wav')])
    raw_files_list = sorted([os.path.join(raw_files, f) for f in os.listdir(raw_files) if f.endswith('.wav')])

    # Ensure the list contains the same amount of files
    assert len(clean_files) == len(raw_files), "The number of clean and raw files must be the same"

    for idx in mic_idx:
        clean_audio, sr = librosa.load(clean_files[idx], sr=sample_r, mono=True)
        raw_audio, sr_raw = librosa.load(raw_files[idx], sr=sample_r, mono=True)

        # NOTE: There is an error here that should disapear when sap pushes
        enhanced_audi = enhance_audio(raw_audio, sample_r, idx)

        separated, srp = enhanced_audi.separated
        natural, srn = enhanced_audi.natural
        suppressed, srs = enhanced_audi.suppressed

        audio_array = [separated, natural, suppressed]





