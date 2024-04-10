import numpy as np
from scipy.io import wavfile

from pathlib import Path

file = wavfile.read("data/raw/noisy_mixture_32ch.wav")

out = Path("data/channels")
if not out.exists():
    out.mkdir()

for i, channel in enumerate(file[1].T):
    wavfile.write(out / f"channel{i}.wav", file[0], channel)