# oticon-audio-explorers-2024

A short description of the repository.

---

## Installation for development
Choose __**ONE**__ of the following:
- Start the devcontainer, then open it in your IDE (best experience, recommended for GPU users).
- Run `pip install -e .[dev]` (recommended for CPU users)


# Data export

**Extracting channels**  
To extract the 32 data channels add the ``.wav`` file to ``models/raw`` and run the following command:
```
python src/modelling/data/utils/export_channels.py
```
The command will save all the channels to ``data/channels``

**Calculating positions**  
To extract the positions of the microphones run the following command:
```
python src/modelling/utils/export_mic_pos.py
```
The command will create ``mic_pos.npy`` in ``src/modelling/utils``