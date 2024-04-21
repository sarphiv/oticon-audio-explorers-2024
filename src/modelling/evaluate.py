from pystoi import stoi
from pesq import pesq
import numpy as np
import os 
import librosa
from modelling.enhance import enhance_audio
from pathlib import Path
from scipy.io import wavfile
import tqdm

def calculate_snr(reference_signal, target_signal):
    # Calculate the Signal to Noise Ratio (SNR) for a single channel
    # SNR = 10 * log10(P_signal / P_noise)
    
    signal_power = np.linalg.norm(target_signal)**2
    noise_power = np.linalg.norm(target_signal - reference_signal)**2
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
    ar_list = ["Seperated Audio", "Natural Audio", "Suppressed Audio", "Raw Audio"]


    # 16000 Hz aligns better with 'wb' mode in PESQ
    default_sr = 16000

    min_length = min(len(true_signal), *(len(x) for x in audio_arrays))
    true_signal = true_signal[:min_length]
    
    normalized_true = true_signal / np.max(np.abs(true_signal))

    for i, enhanced_sig in enumerate(audio_arrays):
        # Trim all signals so they are the same size
        enhanced_sig = enhanced_sig[:min_length]

        metrics = {}
        
        normalized_enhanced = enhanced_sig / np.max(np.abs(enhanced_sig))
        
        # Compute PESQ score
        # pesq(fs, ref, deg, mode='wb', on_error=PesqError.RAISE_EXCEPTION)
        # fs:  integer, sampling rate
        # ref: numpy 1D array, reference audio signal
        # deg: numpy 1D array, degraded audio signal
        
        try:
            pesq_score = pesq(default_sr, normalized_true, normalized_enhanced, 'wb')  # 'wb' is for wide-band
            metrics['PESQ'] = pesq_score
        except Exception as e:
            print(f"PESQ: {e}")

        # Compute STOI score
        # stoi(x, y, fs_sig, extended=False)
        # x (np.ndarray): clean original speech
        # y (np.ndarray): denoised speech
        # fs_sig (int): sampling rate of x and y
        try:
            stoi_score = stoi(normalized_true, normalized_enhanced, default_sr, extended=False)
            metrics['STOI'] = stoi_score
        except Exception as e:
            print(f"STOI: {e}")
   
        # Compute the SNR score
        try:
            snr_score = calculate_snr(normalized_true, normalized_enhanced)
            metrics['SNR'] = snr_score
        except Exception as e:
            print(f"SNR: {e}")

        signal_dict[ar_list[i]] = metrics

    return signal_dict

#mic_idx feed
# np.array([1, 2, 3, 4])
# np.array([32, 12, 30, 14])


def run_metrics(clean_files : Path, raw_files : Path, mic_idx, sample_r = 16000):
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
    clean_files_list = clean_files.glob("*.wav")
    raw_files_list = raw_files.glob("*.wav")
    # clean_files_list = sorted([os.path.join(clean_files, f) for f in os.listdir(clean_files) if f.endswith('.wav')])
    # raw_files_list = sorted([os.path.join(raw_files, f) for f in os.listdir(raw_files) if f.endswith('.wav')])

    # Ensure the list contains the same amount of files
    # assert len(clean_files_list) == len(raw_files_list), "The number of clean and raw files must be the same"

    for i, (clean_file, raw_file) in tqdm.tqdm(enumerate(zip(clean_files_list, raw_files_list))):
        # load using scipy instead of librosa
        
        
        sr_clean, clean_audio = wavfile.read(clean_file)
        sr_raw, raw_audio = wavfile.read(raw_file)

        # NOTE: There is an error here that should disapear when sap pushes
        enhanced_audi = enhance_audio(raw_audio.T, sample_r, microphone_idx = mic_idx)

        separated, srp = enhanced_audi.separated
        natural, srn = enhanced_audi.natural
        suppressed, srs = enhanced_audi.suppressed

        audio_array = [separated, natural, suppressed, raw_audio.mean(axis=1)]

        signal_dict = calculate_metrics(clean_audio.mean(axis=1), audio_array)
        
        metrics_dict[i] = signal_dict
        
        if i > 2:
            break
        
        
    return metrics_dict

if __name__ == "__main__":
    
    clean_files = Path("data/sim_data_less_noise/clean")
    raw_files = Path("data/sim_data_less_noise/clean")
    mic_idx = np.array([1, 2, 3, 4])

    metrics_dict = run_metrics(clean_files, raw_files, mic_idx)
    
    seperated = [0,0,0]
    natural = [0,0,0]
    suppressed = [0,0,0]
    raw = [0,0,0]
    means = {"Seperated Audio": [[],[],[]], "Natural Audio": [[],[],[]], "Suppressed Audio": [[],[],[]], "Raw Audio": [[],[],[]]}
    for i, metrics in enumerate(metrics_dict.values()):
        
        for j, (name, metric) in enumerate(metrics.items()):
            means[name][0].append(metric["PESQ"])
            means[name][1].append(metric["STOI"])
            means[name][2].append(metric["SNR"])
            
    print(" ==== AVERAGE METRICS ==== ")
    print("Signal type |PESQ|STOI|SNR|")
    print(f"Seperated:  {np.round(np.mean(means['Seperated Audio'],axis=1), 3)}")
    print(f"Natural:    {np.round(np.mean(means['Natural Audio'],axis=1), 3)}")
    print(f"Suppressed: {np.round(np.mean(means['Suppressed Audio'],axis=1), 3)}")
    print(f"Raw:        {np.round(np.mean(means['Raw Audio'],axis=1), 3)}")
    
            
    
    
    # print(metrics_dict)