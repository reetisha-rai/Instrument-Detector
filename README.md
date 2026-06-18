# Instrument Detector

Identify every instrument in a song with timestamps — just paste a YouTube or Spotify link.

Built with a CNN-BiLSTM + Self-Attention model trained on IRMAS and OpenMIC-2018.

---

## Project structure

```
instrument-detector/
├── config.py          — all constants (sample rate, classes, paths)
├── requirements.txt
├── .gitignore
└── src/
    ├── model.py        — CNN-BiLSTM + Attention architecture
    ├── preprocessor.py — audio loading, mel-spectrograms, caching
    ├── dataset.py      — IRMAS, OpenMIC-2018, CombinedDataset
    ├── train.py        — training loop, FocalBCE loss
    └── predict.py      — download, inference, timeline output
```

---

## Setup

```bash
pip install -r requirements.txt
# Also install ffmpeg:
# Mac:   brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## Datasets

### IRMAS (~1.5 GB)
Download from https://zenodo.org/record/1290750  
Extract to `data/IRMAS/IRMAS-TrainingData/`

### OpenMIC-2018 (~6 GB)
Download from https://zenodo.org/records/1432913  
Extract to `data/OpenMIC/`

---

## Train

```bash
# Train on both datasets (recommended)
python src/train.py --dataset both --epochs 50

# Train on IRMAS only (faster, fewer instruments)
python src/train.py --dataset irmas --epochs 20

# Custom batch size and learning rate
python src/train.py --dataset both --epochs 50 --batch 128 --lr 1e-3
```

Checkpoint is saved to `models/instrument_detector_best.pt` whenever validation F1 improves.

---

## Predict

```bash
# YouTube link
python src/predict.py --url "https://www.youtube.com/watch?v=..."

# Spotify link
python src/predict.py --url "https://open.spotify.com/track/..."

# Local file
python src/predict.py --file /path/to/song.wav

# Adjust sensitivity (lower = more instruments detected)
python src/predict.py --url "..." --threshold 0.3
```

Results are printed to the terminal and saved as JSON in `outputs/`.

---

## Model

**CNN-BiLSTM + Self-Attention**

- Input: log-mel spectrogram (128 mel bins × 43 frames per 1-second window)
- 3× CNN blocks — extract local frequency/time patterns
- 2× Bidirectional LSTM — capture long-range temporal context
- Self-Attention — weight the most musically active frames
- Sigmoid output — multi-label, 20 instrument classes

**Instrument classes (OpenMIC-2018 taxonomy):**  
accordion, banjo, bass, cello, clarinet, cymbals, drums, flute, guitar,
mallet percussion, mandolin, organ, piano, saxophone, synthesizer,
trombone, trumpet, ukulele, violin, voice

---

## Notes

- The model checkpoint (`.pt` file) is not included in this repo — it is too large for GitHub. Train from scratch using the instructions above, or store it with [Git LFS](https://git-lfs.com/).
- Datasets are not included for the same reason.
- Timestamp resolution is 0.5 seconds (configurable in `config.py`).
