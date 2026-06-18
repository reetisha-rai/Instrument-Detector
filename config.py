"""
config.py — all constants and paths.
Every other file imports from here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR   = BASE_DIR / "tmp"
CACHE_DIR  = DATA_DIR / "cache"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, TEMP_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Instrument classes ────────────────────────────────────────────────────────
INSTRUMENT_CLASSES = [
    'accordion', 'banjo', 'bass', 'cello', 'clarinet',
    'cymbals', 'drums', 'flute', 'guitar', 'mallet_percussion',
    'mandolin', 'organ', 'piano', 'saxophone', 'synthesizer',
    'trombone', 'trumpet', 'ukulele', 'violin', 'voice',
]
NUM_CLASSES = len(INSTRUMENT_CLASSES)

# ── IRMAS class code → instrument name ───────────────────────────────────────
IRMAS_MAP = {
    'cel': 'cello',    'cla': 'clarinet', 'flu': 'flute',
    'gac': 'guitar',   'gel': 'guitar',   'org': 'organ',
    'pia': 'piano',    'sax': 'saxophone','tru': 'trumpet',
    'vio': 'violin',   'voi': 'voice',
}

# ── Audio preprocessing ───────────────────────────────────────────────────────
SAMPLE_RATE    = 22050
N_MELS         = 128
N_FFT          = 2048
HOP_LENGTH     = 512
F_MIN          = 20.0
F_MAX          = 8000.0
WINDOW_SECONDS = 1.0
HOP_SECONDS    = 0.5
FRAMES_PER_WIN = int(WINDOW_SECONDS * SAMPLE_RATE / HOP_LENGTH)

# ── Model architecture ────────────────────────────────────────────────────────
CNN_CHANNELS = [32, 64, 128]
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2
DROPOUT      = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
TRAIN_SPLIT   = 0.8
VAL_SPLIT     = 0.1
EPOCHS        = 50
EARLY_STOP    = 10

# ── Inference ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
MERGE_GAP_SECONDS    = 1.0
MIN_SEGMENT_SECONDS  = 0.5

# ── Checkpoint ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = MODEL_DIR / 'instrument_detector_best.pt'
