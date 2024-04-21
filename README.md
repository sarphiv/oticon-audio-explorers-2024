# oticon-audio-explorers-2024

A short description of the repository.

---

## Installation for development
Choose __**ONE**__ of the following:
- Start the devcontainer, then open it in your IDE (best experience, recommended for GPU users).
- Run `pip install -e .[dev]` (recommended for CPU users)


## Enhancing an audio file
Provide an input directory full of audio files with 32 channels. The output is stored in the specified output directory.
```bash
python -m modelling.enhance --input_dir=data/raw --output_dir=output
```


## Data export

### Extracting channels
To extract the 32 data channels add the ``.wav`` file to ``data/raw`` and run the following command:
```
python src/modelling/data/utils/export_channels.py
```
The command will save all the channels to ``data/channels``

### Calculating positions
To extract the positions of the microphones run the following command:
```python
python -m modelling.utils.export_mic_pos
```
The command will create ``mic_pos.npy`` in ``data/``.
The numpy file is a 32x3 np.ndarray with the shape: `n_mics X 3`.